import unittest

from utils.escalation import EscalationNotifier, ReviewerNotifier
from utils.verification_logger import VerificationLogger

import tempfile


LOCAL_HIGH = {"risk_level": "high", "action": "human_review", "reasoning": "Possible fall detected"}
MODEL_HIGH = {"risk_level": "high", "action": "human_review", "reasoning": "Fall candidate confirmed"}
EVIDENCE = {"frame_path": "frames/risk_high.jpg", "clip_path": "clips/high_risk.mp4"}

VERIFY_RESULT = {
    "agreement": "disagreement",
    "model_decision": {"risk_level": "low", "action": "none", "reasoning": "looks calm"},
    "reason": "risk_level local=high model=low",
    "latency_ms": 812.5,
    "model": "har-vlm",
}


class TestEscalationNotifier(unittest.TestCase):
    def test_digits_normalization(self):
        notifier = EscalationNotifier("+91 98765-43210", "(+1) 555 010 2030")
        self.assertEqual(notifier.whatsapp_number, "919876543210")
        self.assertEqual(notifier.phone_number, "15550102030")

    def test_not_configured_when_empty(self):
        notifier = EscalationNotifier("", "")
        self.assertFalse(notifier.configured)

    def test_escalate_returns_false_when_not_configured(self):
        opened = []
        notifier = EscalationNotifier("", "", opener=opened.append)
        self.assertFalse(notifier.escalate(LOCAL_HIGH, MODEL_HIGH, EVIDENCE))
        self.assertEqual(opened, [])

    def test_build_message_contains_evidence(self):
        notifier = EscalationNotifier("919876543210", "919876543210")
        message = notifier.build_message(LOCAL_HIGH, MODEL_HIGH, EVIDENCE)
        self.assertIn("risk=high", message)
        self.assertIn("Possible fall detected", message)
        self.assertIn("Fall candidate confirmed", message)
        self.assertIn("frames/risk_high.jpg", message)
        self.assertIn("clips/high_risk.mp4", message)
        self.assertIn("not an emergency service", message)

    def test_build_message_without_model_verdict(self):
        notifier = EscalationNotifier("919876543210")
        message = notifier.build_message(LOCAL_HIGH, None, EVIDENCE)
        self.assertIn("Model verdict: unavailable", message)

    def test_build_links(self):
        notifier = EscalationNotifier("919876543210", "9111111111")
        links = notifier.build_links("hello world")
        self.assertEqual(len(links), 2)
        self.assertTrue(links[0].startswith("https://wa.me/919876543210?text="))
        self.assertIn("hello%20world", links[0])
        self.assertEqual(links[1], "tel:9111111111")

    def test_escalate_opens_both_links(self):
        opened = []
        notifier = EscalationNotifier("919876543210", "9111111111", opener=opened.append)
        self.assertTrue(notifier.escalate(LOCAL_HIGH, MODEL_HIGH, EVIDENCE))
        self.assertEqual(len(opened), 2)
        self.assertIn("wa.me/919876543210", opened[0])
        self.assertEqual(opened[1], "tel:9111111111")

    def test_cooldown_blocks_second_escalation(self):
        opened = []
        notifier = EscalationNotifier("919876543210", cooldown_seconds=60, opener=opened.append)
        self.assertTrue(notifier.escalate(LOCAL_HIGH, MODEL_HIGH, EVIDENCE))
        self.assertFalse(notifier.escalate(LOCAL_HIGH, MODEL_HIGH, EVIDENCE))
        self.assertEqual(len(opened), 1)

    def test_dry_run_prints_instead_of_opening(self):
        opened = []
        notifier = EscalationNotifier("919876543210", dry_run=True, opener=opened.append)
        self.assertTrue(notifier.escalate(LOCAL_HIGH, MODEL_HIGH, EVIDENCE))
        self.assertEqual(opened, [])


class TestEvidenceUrl(unittest.TestCase):
    def test_message_uses_photo_url_when_server_configured(self):
        notifier = EscalationNotifier(
            "919876543210", evidence_base_url="http://192.168.1.5:8090"
        )
        message = notifier.build_message(LOCAL_HIGH, MODEL_HIGH, EVIDENCE)
        self.assertIn("Snapshot: http://192.168.1.5:8090/risk_high.jpg", message)
        # The bare path must not appear once a tappable URL exists.
        self.assertNotIn("frames/risk_high.jpg", message)
        self.assertIn("clips/high_risk.mp4", message)

    def test_message_falls_back_to_path_without_server(self):
        notifier = EscalationNotifier("919876543210")
        message = notifier.build_message(LOCAL_HIGH, MODEL_HIGH, EVIDENCE)
        self.assertIn("Screenshot: frames/risk_high.jpg", message)
        self.assertNotIn("Snapshot:", message)

    def test_handles_windows_style_paths(self):
        notifier = EscalationNotifier(
            "919876543210", evidence_base_url="http://192.168.1.5:8090"
        )
        message = notifier.build_message(
            LOCAL_HIGH, MODEL_HIGH, {"frame_path": "frames\\risk_high.jpg"}
        )
        self.assertIn("http://192.168.1.5:8090/risk_high.jpg", message)


class TestReviewerNotifier(unittest.TestCase):
    def test_digits_normalization(self):
        notifier = ReviewerNotifier("+91 99999-88888")
        self.assertEqual(notifier.whatsapp_number, "919999988888")

    def test_not_configured_when_empty(self):
        self.assertFalse(ReviewerNotifier("").configured)

    def test_request_review_returns_false_when_unconfigured(self):
        opened = []
        notifier = ReviewerNotifier("", opener=opened.append)
        self.assertFalse(notifier.request_review(LOCAL_HIGH, VERIFY_RESULT, EVIDENCE))
        self.assertEqual(opened, [])

    def test_request_review_opens_wa_me_link(self):
        opened = []
        notifier = ReviewerNotifier("919999988888", opener=opened.append)
        self.assertTrue(notifier.request_review(LOCAL_HIGH, VERIFY_RESULT, EVIDENCE))
        self.assertEqual(len(opened), 1)
        self.assertIn("wa.me/919999988888", opened[0])
        # The request must tell the reviewer where to answer.
        self.assertIn("approve", opened[0])

    def test_message_includes_both_verdicts_and_reason(self):
        notifier = ReviewerNotifier("919999988888")
        message = notifier.build_message(LOCAL_HIGH, VERIFY_RESULT, EVIDENCE)
        self.assertIn("HUMAN EVALUATION REQUESTED", message)
        self.assertIn("risk=high", message)
        self.assertIn("Verifier verdict: risk=low", message)
        self.assertIn("risk_level local=high model=low", message)

    def test_message_when_verifier_unavailable(self):
        notifier = ReviewerNotifier("919999988888")
        result = {"agreement": "insufficient_output", "model_decision": None, "reason": "call failed"}
        message = notifier.build_message(LOCAL_HIGH, result, EVIDENCE)
        self.assertIn("Verifier verdict: unavailable", message)
        self.assertIn("call failed", message)

    def test_escalation_message_carries_the_agent_alert_text(self):
        # The decision agent's alert_message is what the human reads first.
        notifier = EscalationNotifier("919999988888")
        message = notifier.build_message(
            LOCAL_HIGH, MODEL_HIGH, EVIDENCE,
            alert_message="A person is lying on the corridor floor. Please check now.",
        )
        self.assertIn("lying on the corridor floor", message)
        # The structured verdict lines stay, so nothing is lost if the agent
        # fails to write a message.
        self.assertIn("Local verdict: risk=high", message)
        self.assertIn("Model verdict: risk=high", message)

    def test_escalation_without_alert_text_still_includes_verdicts(self):
        notifier = EscalationNotifier("919999988888")
        message = notifier.build_message(LOCAL_HIGH, MODEL_HIGH, EVIDENCE)
        self.assertIn("Local verdict: risk=high", message)
        self.assertIn("Model verdict: risk=high", message)
        self.assertNotIn("lying on the corridor floor", message)

    def test_reviewer_message_falls_back_to_verify_result_alert(self):
        notifier = ReviewerNotifier("919999988888")
        result = dict(VERIFY_RESULT, alert_message="Someone is on the ground.")
        message = notifier.build_message(LOCAL_HIGH, result, EVIDENCE)
        self.assertIn("Someone is on the ground.", message)

    def test_cooldown_blocks_second_request(self):
        opened = []
        notifier = ReviewerNotifier("919999988888", cooldown_seconds=60, opener=opened.append)
        self.assertTrue(notifier.request_review(LOCAL_HIGH, VERIFY_RESULT, EVIDENCE))
        self.assertFalse(notifier.request_review(LOCAL_HIGH, VERIFY_RESULT, EVIDENCE))
        self.assertEqual(len(opened), 1)

    def test_dry_run_prints_instead_of_opening(self):
        opened = []
        notifier = ReviewerNotifier("919999988888", dry_run=True, opener=opened.append)
        self.assertTrue(notifier.request_review(LOCAL_HIGH, VERIFY_RESULT, EVIDENCE))
        self.assertEqual(opened, [])

    def test_message_uses_photo_url_when_server_configured(self):
        notifier = ReviewerNotifier(
            "919999988888", evidence_base_url="http://192.168.1.5:8090"
        )
        message = notifier.build_message(LOCAL_HIGH, VERIFY_RESULT, EVIDENCE)
        self.assertIn("Snapshot: http://192.168.1.5:8090/risk_high.jpg", message)


class TestVerificationLogger(unittest.TestCase):
    def test_log_appends_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = VerificationLogger(tmp)
            logger.log(42, LOCAL_HIGH, VERIFY_RESULT, "approved", "human_approved")
            content = (logger.log_path).read_text(encoding="utf-8")
            lines = content.strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("disagreement", lines[1])
            self.assertIn("human_approved", lines[1])
            self.assertIn("812.5", lines[1])
            self.assertIn("har-vlm", lines[1])

    def test_header_matches_field_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = VerificationLogger(tmp)
            header = (logger.log_path).read_text(encoding="utf-8").strip().splitlines()[0]
            self.assertEqual(header.split(","), VerificationLogger.FIELDS)

    def test_missing_latency_stays_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = VerificationLogger(tmp)
            result = {"agreement": "agreement", "model_decision": MODEL_HIGH}
            logger.log(1, LOCAL_HIGH, result, "", "agreement")
            row = (logger.log_path).read_text(encoding="utf-8").strip().splitlines()[1]
            # latency and model columns exist but are empty, so the row still
            # has one column per header field.
            self.assertEqual(len(row.split(",")), len(VerificationLogger.FIELDS))

    def test_migrates_legacy_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy_header = [
                "timestamp", "frame_id", "local_risk", "local_action",
                "model_risk", "model_action", "agreement", "human_decision",
                "outcome", "reason",
            ]
            import csv

            legacy_path = f"{tmp}/verification.csv"
            with open(legacy_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(legacy_header)
                writer.writerow(["2026-01-01 00:00:00", 7, "high", "human_review",
                                 "high", "human_review", "agreement", "", "agreement_high", "ok"])

            logger = VerificationLogger(tmp)
            header = (logger.log_path).read_text(encoding="utf-8").strip().splitlines()[0]
            self.assertEqual(header.split(","), VerificationLogger.FIELDS)

            logger.log(42, LOCAL_HIGH, VERIFY_RESULT, "", "agreement")
            rows = (logger.log_path).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 3)
            # Old data survives and stays column-aligned.
            self.assertIn("2026-01-01 00:00:00", rows[1])

    def test_header_created_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            VerificationLogger(tmp)
            VerificationLogger(tmp)
            content = (VerificationLogger(tmp).log_path).read_text(encoding="utf-8")
            self.assertEqual(len(content.strip().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
