"""CSV log for AI event reviews and Telegram delivery outcomes."""
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
        "event_id",
        "alert_type",
        "summary",
        "ai_status",
        "telegram_status",
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
        return verify_result.get("model_decision") or verify_result.get("grok_decision") or {}

    def _append_event_row(self, **values):
        row = [values.get(field, "") for field in self.FIELDS]
        with open(self.log_path, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(row)

    def log(self, frame_id, local_decision, verify_result, human_decision, outcome):
        """Compatibility logger for the previous single-frame verifier path."""
        verify_result = verify_result or {}
        model_decision = self._model_decision(verify_result)
        latency = verify_result.get("latency_ms", "")
        if isinstance(latency, (int, float)):
            latency = f"{latency:.1f}"
        self._append_event_row(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_id=frame_id,
            local_risk=local_decision.get("risk_level", ""),
            local_action=local_decision.get("action", ""),
            model_risk=model_decision.get("risk_level", ""),
            model_action=model_decision.get("action", ""),
            agreement=verify_result.get("agreement", ""),
            human_decision=human_decision,
            outcome=outcome,
            vlm_latency_ms=latency,
            vlm_model=verify_result.get("model", ""),
            reason=verify_result.get("reason", ""),
        )

    def log_event_result(self, result):
        """Append one event review and the resulting Telegram delivery state."""
        local = result.get("local_decision") or {}
        decision = result.get("decision") or {}
        review = result.get("review") or {}
        delivery = result.get("delivery") or {}
        risk_level = decision.get("risk_level", "")
        alert_type = decision.get("alert_type", "")
        review_status = review.get("status", "error")
        if delivery.get("video_sent"):
            telegram_status = "video_sent"
        elif delivery.get("text_sent"):
            telegram_status = "text_sent"
        elif delivery.get("error") == "low risk":
            telegram_status = "skipped_low_risk"
        elif delivery.get("error"):
            telegram_status = "failed"
        else:
            telegram_status = "not_configured"
        latency = review.get("latency_ms", "")
        if isinstance(latency, (int, float)):
            latency = f"{latency:.1f}"

        self._append_event_row(
            timestamp=result.get("timestamp", ""),
            frame_id=result.get("frame_id", ""),
            local_risk=local.get("risk_level", ""),
            local_action=local.get("action", ""),
            model_risk=risk_level,
            model_action=("human_review" if risk_level in {"medium", "high"} or alert_type == "uncertain" else "none"),
            agreement=review_status,
            human_decision="",
            outcome=f"{review_status}:{telegram_status}",
            vlm_latency_ms=latency,
            vlm_model=review.get("model", ""),
            reason=review.get("reason", ""),
            event_id=result.get("event_id", ""),
            alert_type=alert_type,
            summary=decision.get("summary", ""),
            ai_status=review_status,
            telegram_status=telegram_status,
        )