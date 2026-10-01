"""Renewal Analysis Agent: Anthropic SDK tool-calling loop over the contract shim and policy KB.

Usage:  python agents/renewal_agent.py            # auto-picks 5 varied contracts
        python agents/renewal_agent.py C001 C002  # analyse specific contract ids
Requires: pip install anthropic requests chromadb ; ANTHROPIC_API_KEY set ; contract_shim.py running.
"""
import json
import os
import sys

import chromadb
import requests
from anthropic import Anthropic
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
SHIM = "http://localhost:5001"
MODEL = os.environ.get("CRRA_MODEL", "claude-opus-5-5")  # override via env if your account uses another id
CHROMA_PATH = os.environ.get("CRRA_CHROMA_PATH", os.path.join(ROOT, "data", "chroma_db"))  # must match kb_setup.py
COLLECTION = "crra_policy"
MAX_ROUNDS = 5
RECOMMENDATIONS = ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]
CONFIDENCES = ["HIGH", "MEDIUM", "LOW"]

SYSTEM_PROMPT = """You are a contract renewal analyst for an enterprise procurement team.
For the contract you are given, gather evidence with your tools, then call submit_recommendation exactly once.

Process:
1. get_contract to read the contract, including notice_state, utilisation_pct and approval_band.
2. search_policy with a focused query about renewal, notice, utilisation or approval rules relevant to this contract.
3. find_category_overlap for the contract's category to check for vendor overlap or consolidation potential.
4. submit_recommendation.

Rules:
- Base every claim on tool results. Never invent policy text, figures or vendors. Cite the policy source and section in policy_citation; if no relevant policy was found, say so there.
- LOW confidence is a valid and expected answer. If data is missing, the policy is ambiguous, the tools failed, or the evidence points two ways, choose LOW instead of guessing. An honest LOW is better than a confident error.
- Weigh notice_state urgency (INSIDE_WINDOW and EXPIRED limit your options) alongside utilisation and category overlap.
- estimated_annual_impact_inr is signed: positive means annual saving, negative means added annual cost, 0 if unknown.
- Set human_approval_required to true whenever policy requires approval for this contract, confidence is LOW, or you are unsure; when in doubt, true.
- Be concise. You have a strict limit on tool-calling rounds, so request independent tools together."""

TOOLS = [
    {"name": "get_contract",
     "description": "Fetch one contract with derived fields (notice_deadline, notice_state, utilisation_pct, approval_band).",
     "input_schema": {"type": "object", "properties": {"contract_id": {"type": "string"}}, "required": ["contract_id"]}},
    {"name": "search_policy",
     "description": "Semantic search of the procurement policy knowledge base. Returns the best section from each of the top 2 policy files.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "find_category_overlap",
     "description": "Return the vendors, contract count and total value for a spend category, to spot consolidation opportunities.",
     "input_schema": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}},
    {"name": "submit_recommendation",
     "description": "Submit the final recommendation. Call once, after gathering evidence.",
     "input_schema": {"type": "object", "additionalProperties": False, "properties": {
         "contract_id": {"type": "string"},
         "recommendation": {"type": "string", "enum": RECOMMENDATIONS},
         "confidence": {"type": "string", "enum": CONFIDENCES},
         "rationale": {"type": "string"},
         "policy_citation": {"type": "string", "description": "Policy source file and section relied on, or a statement that none applied."},
         "estimated_annual_impact_inr": {"type": "number", "description": "Signed: positive = saving, negative = added cost."},
         "human_approval_required": {"type": "boolean"}},
         "required": ["contract_id", "recommendation", "confidence", "rationale", "policy_citation",
                      "estimated_annual_impact_inr", "human_approval_required"]}},
]

_collection = None


def get_contract(contract_id):
    try:
        r = requests.get(f"{SHIM}/api/contracts/{contract_id}", timeout=10)
    except requests.exceptions.RequestException:
        return {"error": f"Contract service unreachable at {SHIM}. Start mcp_server/contract_shim.py and retry. "
                         "Without contract data, recommend with LOW confidence."}
    if r.status_code == 404:
        return {"error": f"Contract '{contract_id}' not found."}
    return r.json() if r.ok else {"error": f"Contract service returned HTTP {r.status_code}."}


def search_policy(query):
    global _collection
    try:
        if _collection is None:
            _collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
        res = _collection.query(query_texts=[query], n_results=10, include=["documents", "metadatas", "distances"])
    except Exception as e:  # missing collection, bad path, embedding failure
        return {"error": f"Policy search failed: {e}"}
    best = {}  # source file -> (distance, metadata, text)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        meta = meta or {}
        src = meta.get("source", "unknown")
        if src not in best or dist < best[src][0]:
            best[src] = (dist, meta, doc)
    top = sorted(best.values(), key=lambda t: t[0])[:2]
    if not top:
        return {"results": [], "note": "No policy sections matched."}
    return {"results": [{"source": m.get("source", "unknown"),
                         "section": m.get("section") or m.get("heading") or "unknown",
                         "confidence": round(1 / (1 + d), 3),  # monotonic transform of distance, 1 = identical
                         "text": doc} for d, m, doc in top]}


def find_category_overlap(category):
    try:
        r = requests.get(f"{SHIM}/api/categories", timeout=10)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        return {"error": f"Category service unreachable at {SHIM}. Treat overlap as unknown."}
    for entry in r.json()["categories"]:
        if entry["category"].lower() == category.lower():
            return entry
    return {"error": f"No category named '{category}'."}


def submit_recommendation(**kw):
    if kw.get("recommendation") not in RECOMMENDATIONS or kw.get("confidence") not in CONFIDENCES \
            or not isinstance(kw.get("human_approval_required"), bool):
        return {"error": f"Invalid submission. recommendation must be one of {RECOMMENDATIONS}, confidence one of "
                         f"{CONFIDENCES}, human_approval_required a boolean. Resubmit."}
    return {"status": "recorded"}


HANDLERS = {"get_contract": get_contract, "search_policy": search_policy,
            "find_category_overlap": find_category_overlap, "submit_recommendation": submit_recommendation}


def first_text(response):
    """First block with a .text attribute (a thinking block may come first)."""
    return next((b.text for b in response.content if hasattr(b, "text")), "")


def run_agent(client, contract_id):
    messages = [{"role": "user", "content": f"Analyse renewal for contract {contract_id} and submit a recommendation."}]
    for _ in range(MAX_ROUNDS):
        resp = client.messages.create(model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT, tools=TOOLS,
                                      messages=messages, extra_body={"output_config": {"effort": "medium"}})
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return {"contract_id": contract_id, "status": "NO_SUBMISSION", "note": first_text(resp)}
        results, submitted = [], None
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                out = HANDLERS[block.name](**block.input)
            except Exception as e:
                out = {"error": f"{block.name} failed: {e}"}
            if block.name == "submit_recommendation" and "error" not in out:
                submitted = block.input
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(out), "is_error": "error" in out})
        if submitted:
            return {"status": "OK", **submitted}
        messages.append({"role": "user", "content": results})
    return {"contract_id": contract_id, "status": "MAX_ROUNDS_REACHED"}


def pick_test_contracts(n=5):
    try:
        rows = requests.get(f"{SHIM}/api/contracts", timeout=10).json()["contracts"]
    except requests.exceptions.RequestException:
        sys.exit(f"Cannot reach {SHIM}. Start mcp_server/contract_shim.py first.")
    picked, seen = [], set()
    for r in rows:  # first pass: one contract per notice_state for variety
        if r["notice_state"] not in seen:
            seen.add(r["notice_state"])
            picked.append(r["contract_id"])
    picked += [r["contract_id"] for r in rows if r["contract_id"] not in picked]
    return picked[:n]


def print_summary(results):
    cols = [("Contract", 10), ("Recommendation", 15), ("Conf", 7), ("Approval", 9), ("Impact INR", 14), ("Status/Citation", 42)]
    print("\n" + " ".join(f"{h:<{w}}" for h, w in cols))
    print("-" * (sum(w for _, w in cols) + len(cols) - 1))
    for r in results:
        impact = f"{r['estimated_annual_impact_inr']:,.0f}" if "estimated_annual_impact_inr" in r else "-"
        approval = {True: "YES", False: "NO"}.get(r.get("human_approval_required"), "-")
        last = (r.get("policy_citation") or r["status"])[:41]
        row = [str(r["contract_id"]), r.get("recommendation", "-"), r.get("confidence", "-"), approval, impact, last]
        print(" ".join(f"{v:<{w}}" for v, (_, w) in zip(row, cols)))


if __name__ == "__main__":
    client = Anthropic()
    ids = sys.argv[1:] or pick_test_contracts(5)
    results = []
    for cid in ids:
        print(f"Analysing {cid} ...")
        results.append(run_agent(client, cid))
    print_summary(results)