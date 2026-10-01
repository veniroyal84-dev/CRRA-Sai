"""
CRRA Lab C4 — Audit Logger

Every step the supervisor takes on a contract is written down: what the agent
looked up, what it recommended, which policy triggers fired, who approved it and
what the final status was. One JSON object per line (JSON Lines), so the file can
be appended to safely and read back with a few lines of Python or pandas.

Usage:
    from guardrails.audit_logger import AuditLogger

    audit = AuditLogger()
    audit.log("analysis", "contract_fetched", contract_id="CTR-1004", annual_value_inr=6200000)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit_trail.jsonl"


class AuditLogger:
    def __init__(self, log_path: Path | str = DEFAULT_LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, node: str, event: str, contract_id: str | None = None, **details) -> dict:
        """Append one entry to the audit trail, print it, and return it."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "node": node,
            "event": event,
            "contract_id": contract_id,
            "details": details,
        }
        line = json.dumps(entry, default=str)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(f"  [AUDIT] {line}")
        return entry

    def read_all(self) -> list[dict]:
        """Every entry in the trail, oldest first."""
        if not self.log_path.exists():
            return []
        with open(self.log_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]