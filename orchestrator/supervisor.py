"""
CRRA Lab C4 — Supervisor (LangGraph orchestration + human-in-the-loop + audit)

Runs each contract through a four-node graph:

    analysis -> policy_check -> (hitl) -> report

    analysis      fetches the contract from the mock API (Lab C2), then lets the
                  model search the policy KB (Lab C1) and submit a recommendation.
    policy_check  plain Python, no model call: decides whether a human must approve.
    hitl          shows the recommendation and asks a named human to approve it.
    report        sets the final status.

Every node writes to logs/audit_trail.jsonl through guardrails/audit_logger.py.

Prerequisites:
    1. python data/kb_setup.py                  (Lab C1, loads the policy KB)
    2. python mcp_server/contract_shim.py       (Lab C2, in a second terminal)

Run from the project root, with ANTHROPIC_API_KEY set (or in .env):
    python orchestrator/supervisor.py                      # the five test contracts
    python orchestrator/supervisor.py CTR-1004 CTR-1006    # any contract IDs
"""

import sys
from pathlib import Path

# Make the project root importable so "guardrails" resolves when this file is run
# directly as `python orchestrator/supervisor.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import TypedDict

import anthropic
import chromadb
import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from guardrails.audit_logger import AuditLogger

load_dotenv()

MODEL = "claude-opus-5"
MAX_TOKENS = 16000
MAX_ROUNDS = 5  # hard cap on model calls per contract

CONTRACT_API = "http://localhost:5001"
HTTP_TIMEOUT = 10

CHROMA_PATH = PROJECT_ROOT / "chroma_db"
KB_DIR = PROJECT_ROOT / "data" / "kb"
KB_COLLECTION = "crra_policy"

# The C3 test set plus CTR-1010, the one healthy Band A contract, so the
# auto-approve path (no human needed) is exercised too.
TEST_CONTRACTS = ["CTR-1003", "CTR-1012", "CTR-1005", "CTR-1006", "CTR-1004", "CTR-1010"]

audit = AuditLogger()


# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════

class ContractState(TypedDict, total=False):
    contract_id: str
    contract: dict
    recommendation: str | None
    confidence: str | None
    rationale: str
    policy_citation: str
    estimated_annual_impact_inr: int
    hitl_required: bool
    hitl_reason: str
    hitl_approved: bool
    approver: str | None
    final_status: str


# ══════════════════════════════════════════════════════════════
# MODEL CLIENT AND KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════

_client = None


def get_client():
    """Created on first use, so the graph can be imported without an API key."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def extract_text(response) -> str:
    """Text of the first block that has a .text attribute.

    A thinking block can come before the text, so response.content[0].text is
    not safe on this model.
    """
    for block in response.content:
        if hasattr(block, "text"):
            return block.text.strip()
    return ""


_kb = None


def get_kb():
    """Open the crra_policy collection, building it from data/kb if it is missing."""
    global _kb
    if _kb is not None:
        return _kb

    kb_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        _kb = kb_client.get_collection(KB_COLLECTION)
    except Exception:
        print(f"  KB collection '{KB_COLLECTION}' not found in {CHROMA_PATH}, building it from data/kb ...")
        from data.kb_setup import chunk_article  # reuse Lab C1's chunker

        _kb = kb_client.create_collection(KB_COLLECTION, metadata={"hnsw:space": "cosine"})
        ids, docs, metas = [], [], []
        for md in sorted(KB_DIR.glob("*.md")):
            for c in chunk_article(md.read_text(encoding="utf-8"), md.name):
                ids.append(c["id"])
                docs.append(c["document"])
                metas.append(c["metadata"])
        _kb.add(ids=ids, documents=docs, metadatas=metas)
        print(f"  Built KB with {len(ids)} chunks.")
    return _kb


# ══════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "search_policy",
        "description": (
            "Search the BizOps procurement policy knowledge base. Returns the best "
            "matching section from each of the two most relevant policy files, with "
            "source file, section heading, confidence (0-1) and the section text. "
            "Phrase the query as the situation you are deciding about, e.g. "
            "'vendor proposes 18 percent uplift at renewal'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language policy question"}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Submit the final recommendation for the contract. Call exactly once, as the last step.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {
                    "type": "string",
                    "enum": ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"],
                },
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "rationale": {
                    "type": "string",
                    "description": "Two to four sentences. State the specific numbers that drove the decision.",
                },
                "policy_citation": {
                    "type": "string",
                    "description": "Policy file and section relied on, e.g. 'auto_renewal_rules.md > Standard notice windows'.",
                },
                "estimated_annual_impact_inr": {
                    "type": "integer",
                    "description": "Change in annual spend in INR if followed; negative means savings, 0 if none.",
                },
            },
            "required": [
                "recommendation",
                "confidence",
                "rationale",
                "policy_citation",
                "estimated_annual_impact_inr",
            ],
            "additionalProperties": False,
        },
    },
]


def search_policy(query: str) -> dict:
    """Best section per policy file, top two files."""
    kb = get_kb()
    res = kb.query(query_texts=[query], n_results=min(10, kb.count()))

    best_per_source: dict[str, dict] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        src = meta["source"]
        if src not in best_per_source or dist < best_per_source[src]["distance"]:
            best_per_source[src] = {"distance": dist, "heading": meta["heading"], "text": doc}

    ranked = sorted(best_per_source.items(), key=lambda kv: kv[1]["distance"])[:2]
    return {
        "query": query,
        "results": [
            {
                "source": src,
                "section": hit["heading"],
                "confidence": round(1 - hit["distance"], 2),
                "text": hit["text"],
            }
            for src, hit in ranked
        ],
    }


SYSTEM_PROMPT = """You are the Renewal Analysis Agent for Zensar BizOps procurement.

You are given one contract's full record. Decide exactly one recommendation:
- RENEW: keep the contract on its current terms.
- RENEGOTIATE: keep the vendor but push back on price, seats or terms.
- CONSOLIDATE: another vendor in the same category covers this need; merge onto one.
- TERMINATE: the capability itself is no longer required.

How to work:
1. Read notice_state, days_to_notice_deadline, utilisation_pct, proposed_uplift_pct,
   approval_band, auto_renew and owner from the contract record.
2. Call search_policy for the rule that governs your decision (one or two searches).
3. Finish by calling submit_recommendation exactly once. You have at most 5 turns,
   so do not repeat a search you already have.

Rules:
- Base every claim on the contract data and the policy text. Cite the policy file
  and section you relied on in policy_citation.
- Proposed uplift above 15% is never accepted at first offer; that is RENEGOTIATE.
- An absent owner is not by itself a reason to TERMINATE; nobody has confirmed the
  capability is unneeded.
- estimated_annual_impact_inr is the change in annual spend in INR if the
  recommendation is followed: negative for savings, positive for added cost, 0 if none.

Confidence:
- HIGH: the data and a specific policy rule clearly point to one answer.
- MEDIUM: one answer is best, but it rests on an assumption you should name.
- LOW: the data is missing or contradictory, or the policy does not cover the case.
LOW is a valid answer. It sends the contract to a human, which is better than a
confident guess."""


# ══════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════

def fetch_contract(contract_id: str) -> dict:
    try:
        r = requests.get(f"{CONTRACT_API}/api/contracts/{contract_id}", timeout=HTTP_TIMEOUT)
    except requests.exceptions.RequestException as e:
        return {"error": f"Contract API unreachable ({type(e).__name__}). Is contract_shim.py running on port 5001?"}
    if r.status_code == 404:
        return {"error": f"{contract_id} not found"}
    if not r.ok:
        return {"error": f"Contract API returned HTTP {r.status_code}"}
    return r.json()


def _no_recommendation(contract_id: str, contract: dict, reason: str) -> dict:
    audit.log("analysis", "no_recommendation", contract_id, reason=reason)
    return {
        "contract": contract,
        "recommendation": None,
        "confidence": "LOW",
        "rationale": reason,
        "policy_citation": "",
        "estimated_annual_impact_inr": 0,
    }


def analysis_node(state: ContractState) -> dict:
    contract_id = state["contract_id"]
    print(f"\n{'=' * 62}\nANALYSING: {contract_id}\n{'=' * 62}")

    contract = fetch_contract(contract_id)
    if "error" in contract:
        audit.log("analysis", "contract_fetch_failed", contract_id, error=contract["error"])
        return _no_recommendation(contract_id, {}, contract["error"])

    audit.log(
        "analysis",
        "contract_fetched",
        contract_id,
        vendor=contract.get("vendor"),
        annual_value_inr=contract.get("annual_value_inr"),
        approval_band=contract.get("approval_band"),
        notice_state=contract.get("notice_state"),
        owner=contract.get("owner"),
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"Analyse contract {contract_id} and submit your renewal recommendation.\n\n"
                f"Contract record:\n{json.dumps(contract, indent=2)}"
            ),
        }
    ]

    for rounds in range(1, MAX_ROUNDS + 1):
        response = get_client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "medium"},  # temperature is not supported on this model
        )

        text = extract_text(response)
        if text:
            print(f"  [round {rounds}] {text[:200]}")

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_calls:
            return _no_recommendation(
                contract_id, contract, f"Agent stopped ({response.stop_reason}) without submitting a recommendation."
            )

        # Keep the full content, thinking blocks included, so the next request is valid.
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for call in tool_calls:
            if call.name == "submit_recommendation":
                rec = call.input
                audit.log("analysis", "recommendation_submitted", contract_id, rounds=rounds, **rec)
                return {"contract": contract, **rec}

            if call.name == "search_policy":
                try:
                    result = search_policy(call.input["query"])
                    is_error = False
                    audit.log(
                        "analysis",
                        "policy_search",
                        contract_id,
                        query=call.input["query"],
                        hits=[f"{r['source']} > {r['section']} ({r['confidence']})" for r in result["results"]],
                    )
                except Exception as e:  # report to the model rather than crash the graph
                    result = {"error": f"policy KB unavailable: {type(e).__name__}: {e}"}
                    is_error = True
                    audit.log("analysis", "policy_search_failed", contract_id, error=result["error"])
            else:
                result = {"error": f"unknown tool {call.name}"}
                is_error = True

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(result),
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return _no_recommendation(
        contract_id, contract, f"Stopped after {MAX_ROUNDS} rounds with no recommendation."
    )


def policy_check_node(state: ContractState) -> dict:
    """Deterministic guardrail: collect every reason a human must approve."""
    contract = state.get("contract") or {}
    recommendation = state.get("recommendation")
    reasons = []

    if recommendation is None:
        reasons.append("Agent produced no recommendation")

    band = contract.get("approval_band")
    if band in ("B", "C"):
        reasons.append(f"Approval band {band} requires a named human approver")

    if contract.get("notice_state") == "INSIDE_WINDOW":
        reasons.append(
            f"Contract is inside its notice window (notice deadline {contract.get('notice_deadline')})"
        )

    # Sixth trigger: an auto-renewing contract with its notice deadline coming up is
    # the one case where doing nothing is itself a decision. If nobody sends notice
    # by the deadline, the vendor renews for a full term at their price, whatever
    # the value or the recommendation (auto_renewal_rules.md > Why auto-renewal is a risk).
    if contract.get("auto_renew") is True and contract.get("notice_state") == "APPROACHING":
        reasons.append(
            f"Auto-renews unless written notice is sent by {contract.get('notice_deadline')} "
            f"({contract.get('days_to_notice_deadline')} days away). If nobody acts, it renews on "
            f"{contract.get('renewal_date')} for a full term at the vendor's price "
            f"(proposed uplift {contract.get('proposed_uplift_pct')}%) and the leverage to "
            "renegotiate is lost, so a person must decide before the deadline"
        )

    if recommendation == "TERMINATE":
        reasons.append("TERMINATE recommendations always need human sign-off")

    if state.get("confidence") == "LOW":
        reasons.append("Agent confidence is LOW")

    owner = (contract.get("owner") or "").strip().upper()
    if contract and owner in ("", "UNASSIGNED"):
        reasons.append("Contract has no business owner (UNASSIGNED)")

    hitl_required = bool(reasons)
    hitl_reason = "; ".join(reasons)

    print(f"  Policy check: {'HUMAN APPROVAL REQUIRED' if hitl_required else 'auto-approvable'}")
    for r in reasons:
        print(f"    - {r}")

    audit.log(
        "policy_check",
        "policy_checked",
        state["contract_id"],
        hitl_required=hitl_required,
        reasons=reasons,
    )
    return {"hitl_required": hitl_required, "hitl_reason": hitl_reason}


def hitl_node(state: ContractState) -> dict:
    contract_id = state["contract_id"]
    contract = state.get("contract") or {}
    impact = state.get("estimated_annual_impact_inr") or 0

    print(f"\n  +-- HUMAN APPROVAL NEEDED: {contract_id} {'-' * 30}")
    print(f"  | Vendor:          {contract.get('vendor', '?')}  (owner {contract.get('owner', '?')})")
    print(f"  | Recommendation:  {state.get('recommendation') or 'NONE'}   confidence {state.get('confidence')}")
    print(f"  | Impact:          INR {impact:+,}/yr")
    print(f"  | Policy:          {state.get('policy_citation') or '-'}")
    print(f"  | Rationale:       {state.get('rationale', '')}")
    print("  | Why a human is needed:")
    for r in state.get("hitl_reason", "").split("; "):
        print(f"  |   - {r}")
    print(f"  +{'-' * 60}")

    try:
        while True:
            answer = input("  Approve this recommendation? [y/n] ").strip().lower()
            if answer in ("y", "yes", "n", "no"):
                break
            print("  Please answer y or n.")
        approved = answer in ("y", "yes")

        approver = None
        if approved:
            while not approver:
                approver = input("  Approver name: ").strip()
    except EOFError:  # no terminal attached: never approve by default
        print("\n  No input available, treating as rejected.")
        approved, approver = False, None

    audit.log(
        "hitl",
        "human_decision",
        contract_id,
        approved=approved,
        approver=approver,
        recommendation=state.get("recommendation"),
        hitl_reason=state.get("hitl_reason"),
    )
    return {"hitl_approved": approved, "approver": approver}


def report_node(state: ContractState) -> dict:
    contract_id = state["contract_id"]
    rec = state.get("recommendation")

    if not state.get("hitl_required"):
        final_status = f"{rec}_AUTO"
    elif state.get("hitl_approved") and rec:
        final_status = f"{rec}_APPROVED"
    else:
        final_status = "ON_HOLD_REJECTED"

    print(f"  Final status: {final_status}")
    audit.log(
        "report",
        "final_status",
        contract_id,
        final_status=final_status,
        recommendation=rec,
        confidence=state.get("confidence"),
        estimated_annual_impact_inr=state.get("estimated_annual_impact_inr"),
        hitl_required=state.get("hitl_required"),
        approver=state.get("approver"),
    )
    return {"final_status": final_status}


def route_after_policy_check(state: ContractState) -> str:
    return "hitl" if state.get("hitl_required") else "report"


# ══════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(ContractState)
    graph.add_node("analysis", analysis_node)
    graph.add_node("policy_check", policy_check_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "analysis")
    graph.add_edge("analysis", "policy_check")
    graph.add_conditional_edges(
        "policy_check", route_after_policy_check, {"hitl": "hitl", "report": "report"}
    )
    graph.add_edge("hitl", "report")
    graph.add_edge("report", END)
    return graph.compile()


def print_summary(results: list[dict]) -> None:
    print(f"\n\n{'=' * 78}\nPORTFOLIO SUMMARY\n{'=' * 78}")
    print(f"{'Contract':<11}{'Action':<14}{'Conf':<8}{'Final status':<24}{'Approver':<14}Impact INR/yr")
    print("-" * 78)
    for r in results:
        impact = r.get("estimated_annual_impact_inr") or 0
        print(
            f"{r['contract_id']:<11}{r.get('recommendation') or 'NONE':<14}{r.get('confidence') or '-':<8}"
            f"{r.get('final_status', '-'):<24}{r.get('approver') or '-':<14}{impact:+,}"
        )
    print("-" * 78)
    print(f"Audit trail: {audit.log_path}")


def main(contract_ids: list[str]) -> None:
    app = build_graph()
    audit.log("supervisor", "run_started", contract_ids=contract_ids, model=MODEL)

    results = []
    for cid in contract_ids:
        try:
            results.append(app.invoke({"contract_id": cid}))
        except anthropic.APIError as e:
            print(f"  API error on {cid}: {e}")
            audit.log("supervisor", "api_error", cid, error=f"{type(e).__name__}: {e}")
            results.append({"contract_id": cid, "final_status": "ERROR"})

    audit.log("supervisor", "run_finished", processed=len(results))
    print_summary(results)


if __name__ == "__main__":
    main(sys.argv[1:] or TEST_CONTRACTS)
