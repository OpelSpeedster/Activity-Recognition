import unittest
from unittest.mock import patch

import main
from agents.verifier import INSUFFICIENT

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


if __name__ == "__main__":
    unittest.main()
