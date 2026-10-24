"""CSV log for second-opinion verification results.

Each verified decision gets one row with a timestamp, both verdicts, the
comparison status, the human decision (if a review was requested), and the
verifier latency so the sub-2s target can be tracked over time.
"""
import csv
from datetime import datetime
from pathlib import Path


class VerificationLogger:
    FIELDS = [
        "timestamp",
        "frame_id",
        "local_risk",
        "local_action",
        "model_risk",
        "model_action",
        "agreement",
        "human_decision",
        "outcome",
        "vlm_latency_ms",
        "vlm_model",
        "reason",
    ]

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "verification.csv"
        self._ensure_header()

    def _ensure_header(self):
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(self.FIELDS)
            return
        # Migrate files written by the previous 10-column layout so appended
        # rows stay aligned with the header.
        try:
            with open(self.log_path, "r", newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))
        except (OSError, UnicodeDecodeError):
            return
        if not rows or rows[0] == self.FIELDS:
            return
        old_header = rows[0]
        migrated = [self.FIELDS]
        for row in rows[1:]:
            record = dict(zip(old_header, row))
            migrated.append([record.get(name, "") for name in self.FIELDS])
        with open(self.log_path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerows(migrated)

    @staticmethod
    def _model_decision(verify_result):
        # 'model_decision' is current; 'grok_decision' tolerated for old rows.
        return verify_result.get("model_decision") or verify_result.get("grok_decision") or {}

    def log(self, frame_id, local_decision, verify_result, human_decision, outcome):
        verify_result = verify_result or {}
        model_decision = self._model_decision(verify_result)
        latency = verify_result.get("latency_ms", "")
        if isinstance(latency, (int, float)):
            latency = f"{latency:.1f}"
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_id,
            local_decision.get("risk_level", ""),
            local_decision.get("action", ""),
            model_decision.get("risk_level", ""),
            model_decision.get("action", ""),
            verify_result.get("agreement", ""),
            human_decision,
            outcome,
            latency,
            verify_result.get("model", ""),
            verify_result.get("reason", ""),
        ]
        with open(self.log_path, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(row)
