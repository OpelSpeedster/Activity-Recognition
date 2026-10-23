import unittest

from detectors.fight_detector import FightDetector
from eval.synthetic_scenarios import fight_scenario, two_people_passing_scenario, FRAME_SHAPE


class TestFightDetector(unittest.TestCase):
    def test_detects_real_fight(self):
        frames, frame_shape, _ = fight_scenario()
        detector = FightDetector()
        total_fights = []
        for poses in frames:
            fights = detector.update(poses, frame_shape)
            total_fights.extend(fights)
        detected_ids = {f["id"] for f in total_fights}
        self.assertEqual(detected_ids, {1, 2}, "Both participants should be flagged")

    def test_no_false_positive_on_passing(self):
        frames, frame_shape, _ = two_people_passing_scenario()
        detector = FightDetector()
        total_fights = []
        for poses in frames:
            fights = detector.update(poses, frame_shape)
            total_fights.extend(fights)
        self.assertEqual(len(total_fights), 0, "Passing near each other must not trigger a fight")

    def test_single_frame_spike_does_not_trigger(self):
        """A one-frame motion spike (e.g. tracking jitter) should NOT fire
        without sustained min_consec_frames -- this is the debounce fix."""
        detector = FightDetector(min_consec_frames=3, spike_threshold=0.06)
        bbox_a = [100, 100, 150, 250]
        bbox_b = [140, 100, 190, 250]  # close together

        def kp(cx, cy):
            return [(cx, cy, 0.9)] * 17

        poses_still = [
            {"id": 1, "bbox": bbox_a, "keypoints": kp(125, 175)},
            {"id": 2, "bbox": bbox_b, "keypoints": kp(165, 175)},
        ]
        detector.update(poses_still, FRAME_SHAPE)

        # one single large jump then back to still (simulates a 1-frame glitch)
        poses_spike = [
            {"id": 1, "bbox": bbox_a, "keypoints": kp(125 + 40, 175 + 40)},
            {"id": 2, "bbox": bbox_b, "keypoints": kp(165, 175)},
        ]
        fights = detector.update(poses_spike, FRAME_SHAPE)
        detector.update(poses_still, FRAME_SHAPE)
        detector.update(poses_still, FRAME_SHAPE)

        self.assertEqual(len(fights), 0, "A single-frame spike alone must not fire a fight event")

    def test_fight_not_retriggered_every_frame(self):
        frames, frame_shape, _ = fight_scenario()
        detector = FightDetector()
        total_fights = []
        for poses in frames:
            fights = detector.update(poses, frame_shape)
            total_fights.extend(fights)
        # exactly one event per person for the single sustained fight window
        self.assertEqual(len(total_fights), 2)


if __name__ == "__main__":
    unittest.main()
