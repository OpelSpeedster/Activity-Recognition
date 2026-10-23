import unittest

from detectors.fall_detector import FallDetector
from eval.synthetic_scenarios import fall_scenario, normal_walking_scenario, sitting_scenario


class TestFallDetector(unittest.TestCase):
    def test_detects_real_fall(self):
        frames, _, _ = fall_scenario()
        detector = FallDetector()
        total_falls = []
        for poses in frames:
            falls, _ = detector.update(poses)
            total_falls.extend(falls)
        self.assertGreaterEqual(len(total_falls), 1, "Should detect at least one fall event")

    def test_no_false_positive_on_walking(self):
        frames, _, _ = normal_walking_scenario()
        detector = FallDetector()
        total_falls = []
        for poses in frames:
            falls, _ = detector.update(poses)
            total_falls.extend(falls)
        self.assertEqual(len(total_falls), 0, "Normal walking must not trigger a fall")

    def test_no_false_positive_on_sitting(self):
        frames, _, _ = sitting_scenario()
        detector = FallDetector()
        total_falls = []
        for poses in frames:
            falls, _ = detector.update(poses)
            total_falls.extend(falls)
        self.assertEqual(len(total_falls), 0, "Sitting down must not trigger a fall")

    def test_fall_not_retriggered_every_frame_while_down(self):
        """A person who stays down should only fire ONE fall event, not one per frame."""
        frames, _, _ = fall_scenario()
        detector = FallDetector()
        total_falls = []
        for poses in frames:
            falls, _ = detector.update(poses)
            total_falls.extend(falls)
        self.assertEqual(len(total_falls), 1, "Should fire exactly one fall event, not repeat every frame")

    def test_inactivity_detected_while_down(self):
        frames, _, _ = fall_scenario()
        detector = FallDetector()
        total_inactivity = []
        for poses in frames:
            _, inactivity = detector.update(poses)
            total_inactivity.extend(inactivity)
        self.assertGreater(len(total_inactivity), 0, "Should detect inactivity once person is down and still")

    def test_person_removed_on_track_loss(self):
        detector = FallDetector(min_frames=2)
        pose = {"id": 1, "bbox": [0, 0, 10, 10], "keypoints": [(0, 0, 0.9)] * 17}
        detector.update([pose])
        self.assertIn(1, detector.prev_poses)
        detector.update([])  # person disappears
        self.assertNotIn(1, detector.prev_poses)
        self.assertNotIn(1, detector.horizontal_counts)


if __name__ == "__main__":
    unittest.main()
