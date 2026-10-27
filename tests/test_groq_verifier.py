import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from agents.verifier import AGREEMENT, DISAGREEMENT, INSUFFICIENT
from agents.vlm_verifier import (
    GroqVerifier,
    build_context_summary,
    encode_frame_jpeg,
)


def fake_response(content):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def make_verifier():
    verifier = GroqVerifier(
        "groq-test-key",
        vision_model="test-vision",
        decision_model="test-decision",
        base_url="https://example.invalid/v1",
    )
    verifier.vision.client = MagicMock()
    verifier.decision.client = MagicMock()
    return verifier


def make_frame():
    import numpy as np

    # 64x64 BGR image.
    return np.zeros((64, 64, 3), dtype=np.uint8)


FALL_CONTEXT = {
    "frame_id": 42,
    "timestamp": 0.0,
    "crowd": {"count": 3, "density": 0.2, "level": "low"},
    "falls": [{"track_id": 1, "confidence": 0.9}],
    "fights": [],
    "inactivity": [],
    "prolonged_risk": False,
    "recent_events": ["fall"],
}

LOCAL_HIGH = {
    "risk_level": "high",
    "action": "human_review",
    "reasoning": "Possible fall detected",
}

OBSERVATION_JSON = (
    '{"scene": "indoor corridor", "activity": "one person on the floor", '
    '"people_count": 2, "risk_signals": ["person on the ground"], '
    '"confidence": "medium", "notes": "low light"}'
)

HIGH_JSON = (
    '{"risk_level": "high", "action": "human_review", '
    '"reasoning": "person on the ground", '
    '"alert_message": "A person is lying on the corridor floor. Please check now."}'
)

LOW_JSON = (
    '{"risk_level": "low", "action": "none", "reasoning": "everyone upright", '
    '"alert_message": "Nothing concerning is visible."}'
)


def stub(verifier, decision_json=HIGH_JSON, observation_json=OBSERVATION_JSON):
    verifier.vision.client.chat.completions.create.return_value = fake_response(
        observation_json
    )
    verifier.decision.client.chat.completions.create.return_value = fake_response(
        decision_json
    )
    return verifier


class TestGroqVerifierInit(unittest.TestCase):
    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            GroqVerifier("", base_url="https://example.invalid/v1")

    def test_default_models_come_from_the_agentic_layer(self):
        verifier = GroqVerifier("groq-test-key")
        self.assertEqual(verifier.vision.model, "qwen/qwen3.8-27b")
        self.assertEqual(verifier.decision.model, "openai/gpt-oss-120b")
        self.assertEqual(verifier.model, "openai/gpt-oss-120b")

    def test_both_agents_share_one_endpoint(self):
        verifier = make_verifier()
        self.assertEqual(verifier.vision.base_url, "https://example.invalid/v1")
        self.assertEqual(verifier.decision.base_url, "https://example.invalid/v1")


class TestVerify(unittest.TestCase):
    def test_agreement(self):
        result = stub(make_verifier()).verify(make_frame(), FALL_CONTEXT, LOCAL_HIGH)
        self.assertEqual(result["agreement"], AGREEMENT)
        self.assertEqual(result["model_decision"]["risk_level"], "high")
        self.assertGreaterEqual(result["latency_ms"], 0.0)
        self.assertEqual(result["model"], "test-decision")
        self.assertEqual(result["vision_model"], "test-vision")

    def test_disagreement(self):
        result = stub(make_verifier(), decision_json=LOW_JSON).verify(
            make_frame(), FALL_CONTEXT, LOCAL_HIGH
        )
        self.assertEqual(result["agreement"], DISAGREEMENT)

    def test_alert_message_is_surfaced(self):
        result = stub(make_verifier()).verify(make_frame(), FALL_CONTEXT, LOCAL_HIGH)
        self.assertIn("corridor", result["alert_message"])

    def test_observation_is_kept_for_the_logs(self):
        result = stub(make_verifier()).verify(make_frame(), FALL_CONTEXT, LOCAL_HIGH)
        self.assertEqual(result["observation"]["confidence"], "medium")

    def test_invalid_vision_output_is_insufficient(self):
        verifier = make_verifier()
        verifier.vision.client.chat.completions.create.return_value = fake_response(
            "I cannot help with that."
        )
        result = verifier.verify(make_frame(), FALL_CONTEXT, LOCAL_HIGH)
        self.assertEqual(result["agreement"], INSUFFICIENT)
        self.assertIsNone(result["model_decision"])
        # The decision agent must not run on a junk observation.
        verifier.decision.client.chat.completions.create.assert_not_called()

    def test_invalid_decision_output_keeps_the_observation(self):
        verifier = make_verifier()
        verifier.vision.client.chat.completions.create.return_value = fake_response(
            OBSERVATION_JSON
        )
        verifier.decision.client.chat.completions.create.return_value = fake_response(
            "risk_level is critical"
        )
        result = verifier.verify(make_frame(), FALL_CONTEXT, LOCAL_HIGH)
        self.assertEqual(result["agreement"], INSUFFICIENT)
        self.assertIsNone(result["model_decision"])
        self.assertEqual(result["observation"]["confidence"], "medium")

    def test_api_exception_is_insufficient(self):
        verifier = make_verifier()
        verifier.vision.client.chat.completions.create.side_effect = Exception("timeout")
        result = verifier.verify(make_frame(), FALL_CONTEXT, LOCAL_HIGH)
        self.assertEqual(result["agreement"], INSUFFICIENT)
        self.assertIn("timeout", result["reason"])
        self.assertIn("Vision agent", result["reason"])

    def test_decision_exception_is_insufficient(self):
        verifier = make_verifier()
        verifier.vision.client.chat.completions.create.return_value = fake_response(
            OBSERVATION_JSON
        )
        verifier.decision.client.chat.completions.create.side_effect = Exception("rate limit")
        result = verifier.verify(make_frame(), FALL_CONTEXT, LOCAL_HIGH)
        self.assertEqual(result["agreement"], INSUFFICIENT)
        self.assertIn("rate limit", result["reason"])
        self.assertIn("Decision agent", result["reason"])

    def test_unencodable_frame_is_insufficient_without_api_call(self):
        verifier = make_verifier()
        result = verifier.verify(None, FALL_CONTEXT, LOCAL_HIGH)
        self.assertEqual(result["agreement"], INSUFFICIENT)
        verifier.vision.client.chat.completions.create.assert_not_called()

    def test_neither_agent_sees_the_local_decision(self):
        verifier = stub(make_verifier())
        local = dict(LOCAL_HIGH, reasoning="LOCAL_MARKER_XYZ")
        verifier.verify(make_frame(), FALL_CONTEXT, local)

        vision_text = " ".join(
            str(part)
            for part in verifier.vision.client.chat.completions.create.call_args.kwargs["messages"]
        )
        decision_text = " ".join(
            str(part)
            for part in verifier.decision.client.chat.completions.create.call_args.kwargs["messages"]
        )
        self.assertNotIn("LOCAL_MARKER_XYZ", vision_text)
        self.assertNotIn("LOCAL_MARKER_XYZ", decision_text)

    def test_verify_encoded_accepts_pre_encoded_bytes(self):
        result = stub(make_verifier()).verify_encoded(
            encode_frame_jpeg(make_frame()), FALL_CONTEXT, LOCAL_HIGH
        )
        self.assertEqual(result["agreement"], AGREEMENT)


class TestExports(unittest.TestCase):
    def test_helpers_are_re_exported_from_one_place(self):
        # utils/verify_queue.py and older callers import from vlm_verifier.
        self.assertTrue(callable(build_context_summary))
        self.assertTrue(callable(encode_frame_jpeg))


class FakeVision:
    def __init__(self, fail=False):
        self.fail = fail

    def observe_frames_encoded(self, images, context):
        if self.fail:
            raise RuntimeError("vision unavailable")
        self.seen = (images, context)
        return {"scene": "hall", "activity": "person on floor", "risk_signals": ["possible fall"]}


class FakeDecision:
    def classify_event(self, observation, context):
        self.seen = (observation, context)
        return {"alert_type": "possible_fall", "risk_level": "high", "summary": "Person on the floor."}


class TestGroqEventReview(unittest.TestCase):
    def make_verifier(self, vision):
        verifier = GroqVerifier.__new__(GroqVerifier)
        verifier.vision = vision
        verifier.decision = FakeDecision()
        verifier.model = "fake-decision"
        return verifier

    def test_review_event_passes_all_three_images_then_observation(self):
        vision = FakeVision()
        verifier = self.make_verifier(vision)
        context = {"falls": [{"track_id": 7}]}
        result = verifier.review_event_encoded(["A", "B", "C"], context)
        self.assertEqual(vision.seen[0], ["A", "B", "C"])
        self.assertEqual(verifier.decision.seen[0], result["observation"])
        self.assertEqual(result["model_decision"]["alert_type"], "possible_fall")

    def test_review_event_returns_error_if_vision_fails(self):
        verifier = self.make_verifier(FakeVision(fail=True))
        result = verifier.review_event_encoded(["A", "B", "C"], {})
        self.assertEqual(result["status"], "error")

if __name__ == "__main__":
    unittest.main()
