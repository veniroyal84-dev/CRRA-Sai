"""
CRRA Lab C2 — Mock Contract Management API

Stands in for the real BizOps contract system (Ariba / SAP / a spreadsheet on
someone's laptop). The Analysis Agent in Lab C3 calls this instead of a live
system, so the labs are safe to run and give repeatable results.

Run from the project root:
    python mcp_server/contract_shim.py

Then open http://localhost:5001/health in a browser.
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

CSV_PATH = Path(__file__).parent.parent / "data" / "contracts.csv"

# The whole portfolio is analysed against one fixed "today" so that every
# participant sees identical results no matter which day they run the lab.
SIMULATED_TODAY = datetime(2025, 4, 1)

CONTRACTS: list[dict] = []


def load_contracts() -> None:
    """Read the CSV once at startup and enrich each row with derived fields."""
    CONTRACTS.clear()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["annual_value_inr"] = int(row["annual_value_inr"])
            row["notice_days"] = int(row["notice_days"])
            row["seats_purchased"] = int(row["seats_purchased"])
            row["seats_active"] = int(row["seats_active"])
            row["proposed_uplift_pct"] = int(row["proposed_uplift_pct"])
            row["auto_renew"] = row["auto_renew"].strip().upper() == "Y"

            renewal = datetime.strptime(row["renewal_date"], "%Y-%m-%d")
            deadline = renewal - timedelta(days=row["notice_days"])
            row["notice_deadline"] = deadline.strftime("%Y-%m-%d")
            row["days_to_renewal"] = (renewal - SIMULATED_TODAY).days
            row["days_to_notice_deadline"] = (deadline - SIMULATED_TODAY).days

            # Notice state drives most of the policy logic in Lab C3
            if deadline <= SIMULATED_TODAY <= renewal:
                row["notice_state"] = "INSIDE_WINDOW"
            elif 0 < (deadline - SIMULATED_TODAY).days <= 30:
                row["notice_state"] = "APPROACHING"
            elif renewal < SIMULATED_TODAY:
                row["notice_state"] = "EXPIRED"
            else:
                row["notice_state"] = "OPEN"

            # Seat-based contracts only; AMC/support contracts have no seats
            if row["seats_purchased"] > 0:
                row["utilisation_pct"] = round(
                    100 * row["seats_active"] / row["seats_purchased"]
                )
            else:
                row["utilisation_pct"] = None

            # Approval band per the procurement policy
            v = row["annual_value_inr"]
            row["approval_band"] = "A" if v < 1_000_000 else ("B" if v <= 5_000_000 else "C")

            CONTRACTS.append(row)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "mock-contract-api",
            "contracts_loaded": len(CONTRACTS),
            "simulated_today": SIMULATED_TODAY.strftime("%Y-%m-%d"),
        }
    )


@app.get("/api/contracts")
def list_contracts():
    """All contracts, with optional ?category= ?band= ?notice_state= filters."""
    results = CONTRACTS

    category = request.args.get("category")
    if category:
        results = [c for c in results if c["category"].lower() == category.lower()]

    band = request.args.get("band")
    if band:
        results = [c for c in results if c["approval_band"].upper() == band.upper()]

    notice_state = request.args.get("notice_state")
    if notice_state:
        results = [
            c for c in results if c["notice_state"].upper() == notice_state.upper()
        ]

    return jsonify({"count": len(results), "contracts": results})


@app.get("/api/contracts/<contract_id>")
def get_contract(contract_id):
    for c in CONTRACTS:
        if c["contract_id"].upper() == contract_id.upper():
            return jsonify(c)
    return jsonify({"error": f"Contract {contract_id} not found"}), 404


@app.get("/api/contracts/expiring")
def expiring():
    """Contracts renewing within ?days= (default 90) of the simulated today."""
    window = int(request.args.get("days", 90))
    results = [c for c in CONTRACTS if 0 <= c["days_to_renewal"] <= window]
    results.sort(key=lambda c: c["days_to_renewal"])
    return jsonify({"count": len(results), "window_days": window, "contracts": results})


@app.get("/api/categories")
def categories():
    """Vendor counts per category — the starting point for overlap analysis."""
    grouped: dict[str, list[dict]] = {}
    for c in CONTRACTS:
        grouped.setdefault(c["category"], []).append(
            {
                "contract_id": c["contract_id"],
                "vendor": c["vendor"],
                "annual_value_inr": c["annual_value_inr"],
                "utilisation_pct": c["utilisation_pct"],
            }
        )
    summary = [
        {
            "category": cat,
            "vendor_count": len(items),
            "total_annual_value_inr": sum(i["annual_value_inr"] for i in items),
            "vendors": items,
        }
        for cat, items in sorted(grouped.items())
    ]
    return jsonify({"count": len(summary), "categories": summary})


@app.patch("/api/contracts/<contract_id>")
def update_contract(contract_id):
    """In-memory only. Restarting the server resets every change."""
    payload = request.get_json(silent=True) or {}
    for c in CONTRACTS:
        if c["contract_id"].upper() == contract_id.upper():
            for key in ("status", "owner", "proposed_uplift_pct"):
                if key in payload:
                    c[key] = payload[key]
            return jsonify({"updated": True, "contract": c})
    return jsonify({"error": f"Contract {contract_id} not found"}), 404


if __name__ == "__main__":
    load_contracts()
    print("=" * 60)
    print("  Mock Contract API")
    print(f"  {len(CONTRACTS)} contracts loaded from {CSV_PATH.name}")
    print(f"  Simulated today: {SIMULATED_TODAY:%Y-%m-%d}")
    print("  http://localhost:5001/health")
    print("=" * 60)
    app.run(port=5001, debug=False)
