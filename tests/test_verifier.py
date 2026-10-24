import unittest

from agents.verifier import (
    AGREEMENT,
    DISAGREEMENT,
    HUMAN_APPROVED,
    HUMAN_REJECTED,
    INSUFFICIENT,
    compare_decisions,
    decide_outcome,
    needs_human_review,
    parse_verdict,
)


class TestCompareDecisions(unittest.TestCase):
    def test_agreement(self):
        local = {"risk_level": "high", "action": "human_review", "reasoning": "fall"}
        model = {"risk_level": "high", "action": "human_review", "reasoning": "fall"}
        status, _ = compare_decisions(local, model)
        self.assertEqual(status, AGREEMENT)

    def test_disagreement_on_risk(self):
        local = {"risk_level": "medium", "action": "human_review", "reasoning": "fight"}
        model = {"risk_level": "high", "action": "human_review", "reasoning": "fight"}
        status, reason = compare_decisions(local, model)
        self.assertEqual(status, DISAGREEMENT)
        self.assertIn("risk_level", reason)

    def test_disagreement_on_action(self):
        local = {"risk_level": "medium", "action": "alert", "reasoning": "density"}
        model = {"risk_level": "medium", "action": "human_review", "reasoning": "density"}
        status, reason = compare_decisions(local, model)
        self.assertEqual(status, DISAGREEMENT)
        self.assertIn("action", reason)

    def test_insufficient_when_model_missing(self):
        status, _ = compare_decisions({"risk_level": "low", "action": "none"}, None)
        self.assertEqual(status, INSUFFICIENT)

    def test_insufficient_when_model_not_a_dict(self):
        status, _ = compare_decisions({"risk_level": "low", "action": "none"}, "ok")
        self.assertEqual(status, INSUFFICIENT)


class TestNeedsHumanReview(unittest.TestCase):
    def test_agreement_not_reviewed(self):
        self.assertFalse(needs_human_review({"agreement": AGREEMENT}))

    def test_disagreement_needs_review(self):
        self.assertTrue(needs_human_review({"agreement": DISAGREEMENT}))

    def test_insufficient_needs_review(self):
        self.assertTrue(needs_human_review({"agreement": INSUFFICIENT}))

    def test_none_is_false(self):
        self.assertFalse(needs_human_review(None))


class TestDecideOutcome(unittest.TestCase):
    def test_not_verified_high_escalates(self):
        outcome = decide_outcome({"risk_level": "high"}, None)
        self.assertTrue(outcome["escalate"])
        self.assertTrue(outcome["execute"])

    def test_not_verified_medium_does_not_escalate(self):
        outcome = decide_outcome({"risk_level": "medium"}, None)
        self.assertFalse(outcome["escalate"])
        self.assertTrue(outcome["execute"])

    def test_agreement_high_escalates(self):
        result = {"agreement": AGREEMENT, "model_decision": {"risk_level": "high"}}
        outcome = decide_outcome({"risk_level": "high"}, result)
        self.assertTrue(outcome["escalate"])
        self.assertEqual(outcome["outcome"], "agreement_high")

    def test_agreement_low_continues(self):
        result = {"agreement": AGREEMENT, "model_decision": {"risk_level": "low"}}
        outcome = decide_outcome({"risk_level": "low"}, result)
        self.assertFalse(outcome["escalate"])
        self.assertTrue(outcome["execute"])

    def test_disagreement_approved_with_high_escalates(self):
        result = {"agreement": DISAGREEMENT, "model_decision": {"risk_level": "high"}}
        outcome = decide_outcome({"risk_level": "medium"}, result, HUMAN_APPROVED)
        self.assertTrue(outcome["escalate"])
        self.assertTrue(outcome["execute"])

    def test_disagreement_approved_without_high_continues(self):
        result = {"agreement": DISAGREEMENT, "model_decision": {"risk_level": "low"}}
        outcome = decide_outcome({"risk_level": "medium"}, result, HUMAN_APPROVED)
        self.assertFalse(outcome["escalate"])
        self.assertTrue(outcome["execute"])

    def test_disagreement_rejected_skips_everything(self):
        result = {"agreement": DISAGREEMENT, "model_decision": {"risk_level": "high"}}
        outcome = decide_outcome({"risk_level": "high"}, result, HUMAN_REJECTED)
        self.assertFalse(outcome["escalate"])
        self.assertFalse(outcome["execute"])

    def test_insufficient_approved_uses_local_high(self):
        result = {"agreement": INSUFFICIENT, "model_decision": None}
        outcome = decide_outcome({"risk_level": "high"}, result, HUMAN_APPROVED)
        self.assertTrue(outcome["escalate"])
        self.assertTrue(outcome["execute"])

    def test_insufficient_rejected_skips_everything(self):
        result = {"agreement": INSUFFICIENT, "model_decision": None}
        outcome = decide_outcome({"risk_level": "high"}, result, HUMAN_REJECTED)
        self.assertFalse(outcome["escalate"])
        self.assertFalse(outcome["execute"])

    def test_disagreement_without_approval_channel_never_escalates(self):
        result = {"agreement": DISAGREEMENT, "model_decision": {"risk_level": "high"}}
        outcome = decide_outcome({"risk_level": "high"}, result)
        self.assertFalse(outcome["escalate"])
        self.assertTrue(outcome["execute"])


class TestParseVerdict(unittest.TestCase):
    def test_valid(self):
        out = parse_verdict(
            '{"risk_level": "HIGH", "action": "Human_Review", "reasoning": "fall"}'
        )
        self.assertEqual(
            out, {"risk_level": "high", "action": "human_review", "reasoning": "fall"}
        )

    def test_fenced_json(self):
        text = 'Here you go:\n```json\n{"risk_level": "low", "action": "none", "reasoning": "calm"}\n```'
        out = parse_verdict(text)
        self.assertEqual(out["risk_level"], "low")

    def test_json_wrapped_in_prose(self):
        text = 'The snapshot shows a fall. {"risk_level": "high", "action": "human_review", "reasoning": "person on floor"} I hope that helps.'
        out = parse_verdict(text)
        self.assertEqual(out["risk_level"], "high")

    def test_invalid_risk_level(self):
        with self.assertRaises(ValueError):
            parse_verdict('{"risk_level": "critical", "action": "alert", "reasoning": "x"}')

    def test_invalid_action(self):
        with self.assertRaises(ValueError):
            parse_verdict('{"risk_level": "low", "action": "call police", "reasoning": "x"}')

    def test_empty_reasoning(self):
        with self.assertRaises(ValueError):
            parse_verdict('{"risk_level": "low", "action": "none", "reasoning": "  "}')

    def test_non_dict(self):
        with self.assertRaises(ValueError):
            parse_verdict('["not", "a", "dict"]')

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            parse_verdict("I cannot help with that.")


if __name__ == "__main__":
    unittest.main()
