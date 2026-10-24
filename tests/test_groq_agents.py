import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from agents.groq_agents import (
    DECISION_SCHEMA,
    DEFAULT_DECISION_MODEL,
    DEFAULT_VISION_MODEL,
    GROQ_BASE_URL,
    VISION_SCHEMA,
    DecisionAgent,
    VisionAgent,
    build_context_summary,
    encode_frame_jpeg,
    parse_decision,
    parse_observation,
    render_observation,
)


def fake_response(content):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


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

OBSERVATION_JSON = (
    '{"scene": "indoor corridor", "activity": "one person on the floor", '
    '"people_count": 2, "risk_signals": ["person on the ground"], '
    '"confidence": "medium", "notes": "low light"}'
)

DECISION_JSON = (
    '{"risk_level": "high", "action": "human_review", '
    '"reasoning": "person on the ground", '
    '"alert_message": "A person is lying on the corridor floor. Please check now."}'
)

VALID_OBSERVATION = parse_observation(OBSERVATION_JSON)


def make_agent(agent_cls, model):
    agent = agent_cls("groq-test-key", model=model, base_url="https://example.invalid/v1")
    agent.client = MagicMock()
    return agent


class TestDefaults(unittest.TestCase):
    def test_models_are_live_groq_ids(self):
        self.assertEqual(DEFAULT_VISION_MODEL, "qwen/qwen3.8-27b")
        self.assertEqual(DEFAULT_DECISION_MODEL, "openai/gpt-oss-120b")

    def test_base_url_is_groq(self):
        self.assertTrue(GROQ_BASE_URL.startswith("https://api.groq.com/"))


class TestInit(unittest.TestCase):
    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            VisionAgent("", base_url="https://example.invalid/v1")
        with self.assertRaises(ValueError):
            DecisionAgent("", base_url="https://example.invalid/v1")

    def test_base_url_defaults_to_groq(self):
        agent = VisionAgent("groq-test-key")
        self.assertEqual(agent.base_url, GROQ_BASE_URL.rstrip("/"))


class TestEncodeFrame(unittest.TestCase):
    def test_encodes_ndarray(self):
        import numpy as np

        encoded = encode_frame_jpeg(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertTrue(encoded)
        self.assertIsInstance(encoded, str)

    def test_none_returns_empty(self):
        self.assertEqual(encode_frame_jpeg(None), "")

    def test_empty_frame_array_returns_empty(self):
        import numpy as np

        self.assertEqual(encode_frame_jpeg(np.zeros((0, 0, 3), dtype=np.uint8)), "")


class TestBuildContextSummary(unittest.TestCase):
    def test_contains_measurements(self):
        summary = build_context_summary(FALL_CONTEXT)
        self.assertIn("fall", summary.lower())
        self.assertIn("density", summary.lower())
        self.assertIn("People detected: 3", summary)

    def test_excludes_local_decision(self):
        # Neither agent may see the local verdict, or the comparison becomes
        # the model echoing the answer back.
        self.assertNotIn("LOCAL_MARKER_XYZ", build_context_summary(FALL_CONTEXT))


class TestParseObservation(unittest.TestCase):
    def test_valid(self):
        observation = parse_observation(OBSERVATION_JSON)
        self.assertEqual(observation["scene"], "indoor corridor")
        self.assertEqual(observation["people_count"], 2)
        self.assertEqual(observation["risk_signals"], ["person on the ground"])
        self.assertEqual(observation["confidence"], "medium")

    def test_object_fenced_reply_is_recovered(self):
        observation = parse_observation(f"```json\n{OBSERVATION_JSON}\n```")
        self.assertEqual(observation["confidence"], "medium")

    def test_string_risk_signals_become_a_list(self):
        observation = parse_observation(
            '{"scene": "s", "activity": "a", "people_count": 1, '
            '"risk_signals": "fall", "confidence": "high", "notes": ""}'
        )
        self.assertEqual(observation["risk_signals"], ["fall"])

    def test_blank_signals_dropped(self):
        observation = parse_observation(
            '{"scene": "s", "activity": "a", "people_count": 1, '
            '"risk_signals": ["  ", "crowd"], "confidence": "high", "notes": ""}'
        )
        self.assertEqual(observation["risk_signals"], ["crowd"])

    def test_prose_rejected(self):
        with self.assertRaises(ValueError):
            parse_observation("I cannot help with that.")

    def test_bad_confidence_rejected(self):
        with self.assertRaises(ValueError):
            parse_observation(
                '{"scene": "s", "activity": "a", "people_count": 1, '
                '"risk_signals": [], "confidence": "certain", "notes": ""}'
            )


class TestParseDecision(unittest.TestCase):
    def test_valid(self):
        decision = parse_decision(DECISION_JSON)
        self.assertEqual(decision["risk_level"], "high")
        self.assertEqual(decision["action"], "human_review")
        self.assertTrue(decision["alert_message"])

    def test_values_are_case_insensitive(self):
        decision = parse_decision(
            '{"risk_level": "HIGH", "action": "Human_Review", '
            '"reasoning": "r", "alert_message": "m"}'
        )
        self.assertEqual(decision["risk_level"], "high")
        self.assertEqual(decision["action"], "human_review")

    def test_bad_risk_rejected(self):
        with self.assertRaises(ValueError):
            parse_decision(
                '{"risk_level": "critical", "action": "none", '
                '"reasoning": "r", "alert_message": "m"}'
            )

    def test_bad_action_rejected(self):
        with self.assertRaises(ValueError):
            parse_decision(
                '{"risk_level": "low", "action": "call police", '
                '"reasoning": "r", "alert_message": "m"}'
            )

    def test_empty_reasoning_rejected(self):
        with self.assertRaises(ValueError):
            parse_decision(
                '{"risk_level": "low", "action": "none", '
                '"reasoning": "   ", "alert_message": "m"}'
            )


class TestRenderObservation(unittest.TestCase):
    def test_includes_hazards(self):
        text = render_observation(VALID_OBSERVATION)
        self.assertIn("person on the ground", text)
        self.assertIn("indoor corridor", text)
        self.assertIn("medium", text)

    def test_no_hazards_says_none(self):
        text = render_observation(
            {"scene": "s", "activity": "a", "people_count": 4,
             "risk_signals": [], "confidence": "high", "notes": "n"}
        )
        self.assertIn("none observed", text)


class TestVisionAgent(unittest.TestCase):
    def test_messages_carry_image_block(self):
        agent = make_agent(VisionAgent, "test-vision")
        messages = agent.build_messages("QUJD", FALL_CONTEXT)
        self.assertEqual(messages[0]["role"], "system")
        user_content = messages[1]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        image_block = user_content[1]
        self.assertEqual(image_block["type"], "image_url")
        self.assertEqual(
            image_block["image_url"]["url"], "data:image/jpeg;base64,QUJD"
        )

    def test_system_prompt_carries_guardrails(self):
        agent = make_agent(VisionAgent, "test-vision")
        system = agent.build_messages("QUJD", FALL_CONTEXT)[0]["content"]
        self.assertIn("do not infer a person's intent", system)
        self.assertIn("Do not choose a risk level or an action", system)

    def test_observe_returns_validated_observation(self):
        agent = make_agent(VisionAgent, "test-vision")
        agent.client.chat.completions.create.return_value = fake_response(OBSERVATION_JSON)
        observation = agent.observe_encoded("QUJD", FALL_CONTEXT)
        self.assertEqual(observation["confidence"], "medium")

    def test_observe_uses_vision_model_and_advisory_schema(self):
        agent = make_agent(VisionAgent, "test-vision")
        agent.client.chat.completions.create.return_value = fake_response(OBSERVATION_JSON)
        agent.observe_encoded("QUJD", FALL_CONTEXT)

        kwargs = agent.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "test-vision")
        self.assertFalse(kwargs["response_format"]["json_schema"]["strict"])
        self.assertEqual(kwargs["response_format"]["json_schema"]["schema"], VISION_SCHEMA)

    def test_image_really_goes_out_with_the_request(self):
        agent = make_agent(VisionAgent, "test-vision")
        agent.client.chat.completions.create.return_value = fake_response(OBSERVATION_JSON)
        agent.observe_encoded("QUJD", FALL_CONTEXT)

        messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
        image_parts = [
            part for part in messages[1]["content"] if part["type"] == "image_url"
        ]
        self.assertEqual(len(image_parts), 1)
        self.assertTrue(
            image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )


class TestDecisionAgent(unittest.TestCase):
    def test_messages_carry_observation_and_context(self):
        agent = make_agent(DecisionAgent, "test-decision")
        messages = agent.build_messages(VALID_OBSERVATION, FALL_CONTEXT)
        user_text = messages[1]["content"]
        self.assertIn("person on the ground", user_text)
        self.assertIn("People detected: 3", user_text)

    def test_system_prompt_carries_guardrails(self):
        agent = make_agent(DecisionAgent, "test-decision")
        system = agent.build_messages(VALID_OBSERVATION, FALL_CONTEXT)[0]["content"]
        self.assertIn("Never recommend calling or dispatching emergency services", system)
        self.assertIn("not shown the on-site system's own verdict", system)
        self.assertIn("alert_message", system)

    def test_decide_returns_alert_message(self):
        agent = make_agent(DecisionAgent, "test-decision")
        agent.client.chat.completions.create.return_value = fake_response(DECISION_JSON)
        decision = agent.decide(VALID_OBSERVATION, FALL_CONTEXT)
        self.assertEqual(decision["risk_level"], "high")
        self.assertIn("corridor", decision["alert_message"])

    def test_decide_uses_strict_schema(self):
        agent = make_agent(DecisionAgent, "test-decision")
        agent.client.chat.completions.create.return_value = fake_response(DECISION_JSON)
        agent.decide(VALID_OBSERVATION, FALL_CONTEXT)

        kwargs = agent.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "test-decision")
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])
        self.assertEqual(kwargs["response_format"]["json_schema"]["schema"], DECISION_SCHEMA)

    def test_decision_never_sees_the_local_verdict(self):
        agent = make_agent(DecisionAgent, "test-decision")
        agent.client.chat.completions.create.return_value = fake_response(DECISION_JSON)
        agent.decide(VALID_OBSERVATION, FALL_CONTEXT)

        user_text = agent.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertNotIn("LOCAL_MARKER_XYZ", user_text)


if __name__ == "__main__":
    unittest.main()
