import csv
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from unittest.mock import patch

import main
from agents.rule_agent import RuleBasedAgent
from agents.verifier import INSUFFICIENT
from utils.telegram_notifier import TelegramNotifier
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
        context["falls"] = [{"id": 7}]
        self.assertEqual(main.local_candidate_key(context), ("possible_fall", 7))

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



class FakeEventLoopVideo:
    fps = 2

    def __init__(self):
        self.frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(9)]
        self.read_count = 0
        self.released = False

    def get_frame(self):
        if self.read_count >= len(self.frames):
            return None
        frame = self.frames[self.read_count]
        self.read_count += 1
        return frame

    def release(self):
        self.released = True


class FakePipelineReviewLogger:
    instance = None

    def __init__(self, log_dir):
        self.rows = []
        type(self).instance = self

    def log_event_result(self, result):
        self.rows.append(result)


class FakeGroqEventReviewer:
    def __init__(self):
        self.image_counts = []

    def review_event_encoded(self, images, context):
        self.image_counts.append(len(images))
        return {
            "status": "ok",
            "model_decision": {
                "alert_type": "possible_fall",
                "risk_level": "high",
                "summary": "A person appears to be on the ground.",
            },
            "reason": "",
            "latency_ms": 25.0,
            "model": "fake-gpt-oss",
        }


class FakeTelegramEventNotifier:
    def __init__(self):
        self.calls = []

    def send_event(self, decision, clip_path, timestamp):
        caption = TelegramNotifier.build_caption(decision, timestamp)
        self.calls.append((decision, clip_path, caption))
        return {"video_sent": True, "text_sent": False, "error": ""}


class TestMainEventLoop(unittest.TestCase):
    def test_rule_candidate_runs_fake_groq_then_sends_clip_to_telegram(self):
        video = FakeEventLoopVideo()
        verifier = FakeGroqEventReviewer()
        notifier = FakeTelegramEventNotifier()
        event_logger = FakeEventLogger()
        frame_number = 0

        def fake_process_frame(frame, pipeline, clip_buffer):
            nonlocal frame_number
            frame_number += 1
            falls = [{"id": 7}] if frame_number == 5 else []
            events = [{"type": "fall", "data": falls[0]}] if falls else []
            density = {"count": 1, "density": 0.1, "level": "low"}
            return [], [], events, density, falls, [], []

        with TemporaryDirectory() as directory:
            root = Path(directory)
            def fake_write_clip(frames, event_id):
                clip_path = root / "clips" / "event.mp4"
                clip_path.write_bytes(b"fake-mp4")
                return str(clip_path)

            args = SimpleNamespace(
                source="webcam", camera=0, agent="rule", device=None,
                model=None, imgsz=None, no_display=True,
            )
            with patch.object(main, "parse_args", return_value=args), \
                 patch.object(main, "resolve_pose_profile", return_value=("fake", "cpu", 32)), \
                 patch.object(main, "VideoSource") as source_cls, \
                 patch.object(main, "build_pipeline", return_value=()), \
                 patch.object(main, "process_frame", side_effect=fake_process_frame), \
                 patch.object(main, "EventLogger", return_value=event_logger), \
                 patch.object(main, "VerificationLogger", FakePipelineReviewLogger), \
                 patch.object(main, "build_judge", return_value=verifier), \
                 patch.object(main, "build_telegram_notifier", return_value=notifier), \
                 patch.object(main, "LOG_DIR", root / "logs"), \
                 patch.object(main, "FRAME_DIR", root / "frames"), \
                 patch.object(main, "CLIP_DIR", root / "clips"), \
                 patch.object(main, "CLIP_SECONDS", 4), \
                 patch.object(main, "DISPLAY_WINDOW", False), \
                 patch.object(main, "SAVE_SCREENSHOT_ON_HIGH_RISK", False), \
                 patch.object(main, "AGENT_CALL_EVERY_N_FRAMES", 100), \
                 patch.object(main.ClipBuffer, "_write_event_clip", side_effect=fake_write_clip):
                source_cls.from_webcam.return_value = video
                main.main()
                clip_exists = Path(notifier.calls[0][1]).is_file()

        self.assertEqual(video.read_count, 9)
        self.assertTrue(video.released)
        self.assertEqual(verifier.image_counts, [3])
        self.assertEqual(len(notifier.calls), 1)
        decision, clip_path, caption = notifier.calls[0]
        self.assertTrue(clip_path.endswith("event.mp4"))
        self.assertTrue(clip_exists)
        self.assertIn("possible_fall", decision["alert_type"])
        self.assertIn("Alert type: possible_fall", caption)
        self.assertIn("Risk: high", caption)
        self.assertIn("A person appears to be on the ground.", caption)
        self.assertEqual(len(FakePipelineReviewLogger.instance.rows), 1)
        self.assertEqual(FakePipelineReviewLogger.instance.rows[0]["frame_id"], 5)


if __name__ == "__main__":
    unittest.main()
