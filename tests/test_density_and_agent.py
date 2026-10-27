import unittest

from agents.rule_agent import RuleBasedAgent
from detectors.density_estimator import DensityEstimator

FRAME_SHAPE = (1000, 1000, 3)  # area = 1,000,000


class TestDensityEstimator(unittest.TestCase):
    def test_low_density(self):
        est = DensityEstimator(area_scale=100000, medium_threshold=8, high_threshold=15)
        result = est.estimate([{}] * 2, FRAME_SHAPE)  # density = 2/10 = 0.2
        self.assertEqual(result["level"], "low")
        self.assertEqual(result["count"], 2)

    def test_medium_density(self):
        est = DensityEstimator(area_scale=100000, medium_threshold=8, high_threshold=15)
        result = est.estimate([{}] * 90, FRAME_SHAPE)  # density = 9
        self.assertEqual(result["level"], "medium")

    def test_high_density(self):
        est = DensityEstimator(area_scale=100000, medium_threshold=8, high_threshold=15)
        result = est.estimate([{}] * 160, FRAME_SHAPE)  # density = 16
        self.assertEqual(result["level"], "high")


class TestRuleBasedAgent(unittest.TestCase):
    def setUp(self):
        self.agent = RuleBasedAgent()

    def test_fall_requests_human_review(self):
        context = {"falls": [{"id": 1}], "fights": [], "crowd": {"level": "low"}, "prolonged_risk": False}
        decision = self.agent.analyze(context)
        self.assertEqual(decision["action"], "human_review")
        self.assertEqual(decision["risk_level"], "high")

    def test_fight_in_dense_crowd_requests_human_review(self):
        context = {"falls": [], "fights": [{"id": 1}], "crowd": {"level": "high"}, "prolonged_risk": False}
        decision = self.agent.analyze(context)
        self.assertEqual(decision["action"], "human_review")

    def test_fight_alone_requests_human_review(self):
        context = {"falls": [], "fights": [{"id": 1}], "crowd": {"level": "low"}, "prolonged_risk": False}
        decision = self.agent.analyze(context)
        self.assertEqual(decision["action"], "human_review")
        self.assertEqual(decision["risk_level"], "medium")

    def test_medium_density_requests_review(self):
        context = {"falls": [], "fights": [], "crowd": {"level": "medium"}, "inactivity": [], "prolonged_risk": False}
        decision = self.agent.analyze(context)
        self.assertEqual(decision["action"], "alert")
        self.assertEqual(decision["risk_level"], "medium")

    def test_inactivity_requests_human_review(self):
        context = {"falls": [], "fights": [], "crowd": {"level": "low"}, "inactivity": [{"track_id": 5}], "prolonged_risk": False}
        decision = self.agent.analyze(context)
        self.assertEqual(decision["action"], "human_review")
        self.assertEqual(decision["risk_level"], "medium")
    def test_no_incidents_is_low_risk(self):
        context = {"falls": [], "fights": [], "crowd": {"level": "low"}, "prolonged_risk": False}
        decision = self.agent.analyze(context)
        self.assertEqual(decision["risk_level"], "low")
        self.assertEqual(decision["action"], "none")

    def test_fall_takes_priority_over_fight(self):
        """When two candidates appear together, send the event for review."""
        context = {
            "falls": [{"id": 1}],
            "fights": [{"id": 2}],
            "crowd": {"level": "high"},
            "prolonged_risk": False,
        }
        decision = self.agent.analyze(context)
        self.assertEqual(decision["action"], "human_review")


if __name__ == "__main__":
    unittest.main()
