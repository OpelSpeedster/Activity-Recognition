import csv
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main
from agents.rule_agent import RuleBasedAgent
from agents.verifier import INSUFFICIENT
from utils.verification_logger import VerificationLogger

CONTEXT = {"crowd": {"count": 2, "density": 0.1, "level": "low"}, "falls": []}
EVIDENCE = {"frame_path": "frames/a.jpg", "clip_path": ""}

LOCAL_HIGH = {
    "risk_level": "high",
    "action": "human_review",
    "reasoning": "Possible fall detected",
}
LOCAL_LOW = {"risk_level": "low", "action": "none", "reasoning": "No risk detected"}


class FakeNotifier:
    configured = True

    def __init__(self):
        self.escalations = []
        self.alert_messages = []

    def escalate(self, decision, model_decision, evidence, alert_message=""):
        self.escalations.append((decision, model_decision, evidence))
        self.alert_messages.append(alert_message)
        return True


class FakeActionExecutor:
    def __init__(self):
        self.executed = []

    def execute(self, decision, frame=None):
        self.executed.append(decision)


class FakeEventLogger:
    def __init__(self):
        self.events = []

    def log_event(self, etype, level, message, frame=None):
        self.events.append((etype, level, message))


class FakeVerificationLogger:
    def __init__(self):
        self.rows = []

    def log(self, frame_id, local_decision, verify_result, human_decision, outcome):
        self.rows.append((frame_id, outcome, human_decision))


def make_deps():
    return (
        FakeNotifier(),
        FakeActionExecutor(),
        FakeEventLogger(),
        FakeVerificationLogger(),
    )


class TestUnverifiedDecision(unittest.TestCase):
    """Verification off/unconfigured must not silence alerts."""

    def test_high_risk_still_escalates_and_executes(self):
        notifier, executor, event_logger, _ = make_deps()
        with patch.object(main, "ESCALATE_ON_AGREED_HIGH", True):
            main.handle_unverified_decision(
                dict(LOCAL_HIGH), EVIDENCE, None, notifier, executor, event_logger
            )
        self.assertEqual(len(notifier.escalations), 1)
        self.assertEqual(len(executor.executed), 1)
        self.assertEqual(event_logger.events[0][0], "escalation")

    def test_high_risk_respects_escalation_gate(self):
        notifier, executor, _, _ = make_deps()
        with patch.object(main, "ESCALATE_ON_AGREED_HIGH", False):
            main.handle_unverified_decision(
                dict(LOCAL_HIGH), EVIDENCE, None, notifier, executor, FakeEventLogger()
            )
        self.assertEqual(notifier.escalations, [])
        # Still executes: the gate only silences escalation, not the action.
        self.assertEqual(len(executor.executed), 1)

    def test_low_risk_does_not_escalate(self):
        notifier, executor, _, _ = make_deps()
        with patch.object(main, "ESCALATE_ON_AGREED_HIGH", True):
            main.handle_unverified_decision(
                dict(LOCAL_LOW), EVIDENCE, None, notifier, executor, FakeEventLogger()
            )
        self.assertEqual(notifier.escalations, [])
        self.assertEqual(len(executor.executed), 1)


class TestHandleVerifyResult(unittest.TestCase):
    def test_agreement_high_risk_escalates_without_prompt(self):
        notifier, executor, event_logger, vlog = make_deps()
        verify_result = {
            "agreement": "agreement",
            "model_decision": dict(LOCAL_HIGH),
            "reason": "match",
            "latency_ms": 900.0,
            "model": "har-judge",
        }
        with patch.object(main, "HITL_PROMPT_ENABLED", True), \
             patch.object(main, "ESCALATE_ON_AGREED_HIGH", True), \
             patch.object(main, "VERIFY_LATENCY_TARGET_MS", 4000):
            main.handle_verify_result(
                verify_result, dict(LOCAL_HIGH), EVIDENCE, 11,
                notifier, FakeNotifier(), executor, event_logger, vlog,
            )
        self.assertEqual(len(notifier.escalations), 1)
        self.assertEqual(len(executor.executed), 1)
        self.assertEqual(vlog.rows, [(11, "agreement_high", "")])

    def test_latency_warning_fires_over_target(self):
        notifier, executor, event_logger, vlog = make_deps()
        verify_result = {
            "agreement": "agreement",
            "model_decision": dict(LOCAL_HIGH),
            "reason": "match",
            "latency_ms": 5000.0,
            "model": "har-judge",
        }
        with patch.object(main, "HITL_PROMPT_ENABLED", False), \
             patch.object(main, "ESCALATE_ON_AGREED_HIGH", True), \
             patch.object(main, "VERIFY_LATENCY_TARGET_MS", 4000):
            main.handle_verify_result(
                verify_result, dict(LOCAL_HIGH), EVIDENCE, 12,
                notifier, FakeNotifier(), executor, event_logger, vlog,
            )
        self.assertEqual(vlog.rows[0][1], "agreement_high")

    def test_disagreement_without_hitl_never_escalates(self):
        notifier, executor, event_logger, vlog = make_deps()
        verify_result = {
            "agreement": "disagreement",
            "model_decision": {"risk_level": "low", "action": "none", "reasoning": "calm"},
            "reason": "risk_level local=high model=low",
            "latency_ms": 800.0,
            "model": "har-judge",
        }
        with patch.object(main, "HITL_PROMPT_ENABLED", False), \
             patch.object(main, "ESCALATE_ON_AGREED_HIGH", True):
            main.handle_verify_result(
                verify_result, dict(LOCAL_HIGH), EVIDENCE, 13,
                notifier, FakeNotifier(), executor, event_logger, vlog,
            )
        # An unreviewed mismatch must not notify the authority on its own.
        self.assertEqual(notifier.escalations, [])
        self.assertEqual(len(executor.executed), 1)
        self.assertEqual(vlog.rows[0][1], "disagreement")

    def test_insufficient_output_is_logged(self):
        _, executor, _, vlog = make_deps()
        verify_result = {
            "agreement": INSUFFICIENT,
            "model_decision": None,
            "reason": "AI judge call failed: timeout",
            "latency_ms": 6000.0,
            "model": "har-judge",
        }
        with patch.object(main, "HITL_PROMPT_ENABLED", False):
            main.handle_verify_result(
                verify_result, dict(LOCAL_HIGH), EVIDENCE, 14,
                FakeNotifier(), FakeNotifier(), executor, FakeEventLogger(), vlog,
            )
        self.assertEqual(vlog.rows[0][1], INSUFFICIENT)


class TestShouldVerify(unittest.TestCase):
    """The agentic layer only fires on sustained high risk.

    Everything else -- medium risk, a brief spike, no risk -- runs the normal
    local path with no network call.
    """

    def test_no_judge_never_verifies(self):
        self.assertFalse(main.should_verify(None, LOCAL_HIGH, 99.0))

    def test_low_risk_never_verifies(self):
        self.assertFalse(main.should_verify(object(), LOCAL_LOW, 99.0))

    def test_medium_risk_never_verifies(self):
        medium = {"risk_level": "medium", "action": "alert", "reasoning": "density"}
        self.assertFalse(main.should_verify(object(), medium, 99.0))

    def test_high_risk_barely_started_is_skipped(self):
        # One frame of high risk ~= 1/30s, nowhere near the 2s gate.
        self.assertFalse(main.should_verify(object(), LOCAL_HIGH, 1 / 30))

    def test_high_risk_just_under_threshold_is_skipped(self):
        with patch.object(main, "VERIFY_MIN_HIGH_RISK_SECONDS", 2.0):
            self.assertFalse(main.should_verify(object(), LOCAL_HIGH, 1.99))

    def test_high_risk_at_threshold_verifies(self):
        with patch.object(main, "VERIFY_MIN_HIGH_RISK_SECONDS", 2.0):
            self.assertTrue(main.should_verify(object(), LOCAL_HIGH, 2.0))

    def test_high_risk_long_past_threshold_verifies(self):
        self.assertTrue(main.should_verify(object(), LOCAL_HIGH, 30.0))

    def test_threshold_is_configurable(self):
        # A tighter gate verifies sooner; a looser one holds out longer.
        with patch.object(main, "VERIFY_MIN_HIGH_RISK_SECONDS", 0.5):
            self.assertTrue(main.should_verify(object(), LOCAL_HIGH, 0.5))
        with patch.object(main, "VERIFY_MIN_HIGH_RISK_SECONDS", 0.5):
            self.assertFalse(main.should_verify(object(), LOCAL_HIGH, 0.49))
        with patch.object(main, "VERIFY_MIN_HIGH_RISK_SECONDS", 10.0):
            self.assertFalse(main.should_verify(object(), LOCAL_HIGH, 5.0))


class TestJudgeSelection(unittest.TestCase):
    """The judge is the Groq two-agent verifier; None means no second opinion."""

    def test_disabled_returns_none(self):
        with patch.object(main, "VERIFY_ENABLED", False):
            self.assertIsNone(main.build_judge())

    def test_missing_groq_key_returns_none(self):
        with patch.object(main, "VERIFY_ENABLED", True), \
             patch.object(main, "GROQ_API_KEY", ""):
            self.assertIsNone(main.build_judge())

    def test_builds_groq_verifier_with_configured_models(self):
        with patch.object(main, "VERIFY_ENABLED", True), \
             patch.object(main, "GROQ_API_KEY", "groq-a.b"), \
             patch.object(main, "VISION_MODEL", "test-vision"), \
             patch.object(main, "DECISION_MODEL", "test-decision"):
            judge = main.build_judge()
        self.assertEqual(type(judge).__name__, "GroqVerifier")
        self.assertEqual(judge.vision.model, "test-vision")
        self.assertEqual(judge.decision.model, "test-decision")
        self.assertEqual(judge.model, "test-decision")

    def test_constructor_failure_returns_none(self):
        # A bad key shape must not kill startup; verification just drops out.
        with patch.object(main, "VERIFY_ENABLED", True), \
             patch.object(main, "GROQ_API_KEY", "groq-a.b"), \
             patch("main.GroqVerifier", side_effect=ValueError("bad key")):
            self.assertIsNone(main.build_judge())

    def test_alert_message_reaches_the_escalation(self):
        notifier, executor, event_logger, vlog = make_deps()
        verify_result = {
            "agreement": "agreement",
            "model_decision": dict(LOCAL_HIGH),
            "reason": "match",
            "latency_ms": 400.0,
            "model": "test-decision",
            "alert_message": "A person is on the floor. Please check now.",
        }
        with patch.object(main, "HITL_PROMPT_ENABLED", False), \
             patch.object(main, "ESCALATE_ON_AGREED_HIGH", True):
            main.handle_verify_result(
                verify_result, dict(LOCAL_HIGH), EVIDENCE, 21,
                notifier, FakeNotifier(), executor, event_logger, vlog,
            )
        self.assertEqual(notifier.alert_messages[0], verify_result["alert_message"])

    def test_missing_alert_message_falls_back_to_empty(self):
        notifier, executor, event_logger, vlog = make_deps()
        verify_result = {
            "agreement": "agreement",
            "model_decision": dict(LOCAL_HIGH),
            "reason": "match",
            "latency_ms": 400.0,
            "model": "test-decision",
        }
        with patch.object(main, "HITL_PROMPT_ENABLED", False), \
             patch.object(main, "ESCALATE_ON_AGREED_HIGH", True):
            main.handle_verify_result(
                verify_result, dict(LOCAL_HIGH), EVIDENCE, 22,
                notifier, FakeNotifier(), executor, event_logger, vlog,
            )
        self.assertEqual(notifier.alert_messages[0], "")



class RecordingPipeline:
    def __init__(self):
        self.submitted = []

    def submit(self, incident):
        self.submitted.append(incident)
        return True


class TestRuleEventGate(unittest.TestCase):
    def test_none_rule_action_does_not_submit(self):
        pipeline = RecordingPipeline()
        incident = {"event_id": "event-1"}
        self.assertFalse(main.submit_rule_candidate(LOCAL_LOW, incident, pipeline))
        self.assertEqual(pipeline.submitted, [])

    def test_non_none_rule_action_submits_candidate(self):
        pipeline = RecordingPipeline()
        incident = {"event_id": "event-1"}
        self.assertTrue(main.submit_rule_candidate(LOCAL_HIGH, incident, pipeline))
        self.assertEqual(pipeline.submitted, [incident])

    def test_same_event_is_deduplicated_within_cooldown(self):
        last_seen = {}
        key = ("possible_fall", "track-7")
        self.assertTrue(main.is_new_event(key, 10.0, last_seen, cooldown=5.0))
        self.assertFalse(main.is_new_event(key, 12.0, last_seen, cooldown=5.0))
        self.assertTrue(main.is_new_event(key, 16.0, last_seen, cooldown=5.0))

    def test_local_candidate_key_matches_deterministic_priority(self):
        context = {
            "falls": [],
            "fights": [],
            "crowd": {"level": "medium"},
            "inactivity": [{"track_id": 8}],
        }
        self.assertEqual(main.local_candidate_key(context), ("crowd_density", None))
        context["crowd"]["level"] = "low"
        self.assertEqual(main.local_candidate_key(context), ("inactivity", 8))

    def test_inactivity_rule_creates_a_deterministic_candidate(self):
        decision = RuleBasedAgent().analyze({
            "falls": [], "fights": [], "crowd": {"level": "low"},
            "inactivity": [{"track_id": 5}], "prolonged_risk": False,
        })
        self.assertEqual(decision["action"], "human_review")


class TestEventVerificationLogging(unittest.TestCase):
    def test_event_result_and_legacy_rows_use_the_expanded_header(self):
        with TemporaryDirectory() as directory:
            logger = VerificationLogger(directory)
            logger.log_event_result({
                "event_id": "possible_fall-7-10",
                "frame_id": 10,
                "timestamp": "2026-10-27 10:00:00",
                "local_decision": LOCAL_HIGH,
                "review": {"status": "ok", "latency_ms": 12.3, "model": "fake-model", "reason": ""},
                "decision": {"alert_type": "possible_fall", "risk_level": "high", "summary": "Person on floor."},
                "delivery": {"video_sent": True, "text_sent": False, "error": ""},
            })
            logger.log(11, LOCAL_HIGH, {"agreement": "agreement", "model_decision": LOCAL_HIGH}, "", "agreement_high")
            with open(logger.log_path, "r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["alert_type"], "possible_fall")
        self.assertEqual(rows[0]["telegram_status"], "video_sent")
        self.assertEqual(rows[0]["summary"], "Person on floor.")
        self.assertEqual(rows[1]["outcome"], "agreement_high")
        self.assertTrue(all(len(row) == len(VerificationLogger.FIELDS) for row in rows))

if __name__ == "__main__":
    unittest.main()
