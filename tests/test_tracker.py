import unittest

from utils.simple_tracker import SimpleTracker

FRAME_SHAPE = (720, 1280, 3)


class TestSimpleTracker(unittest.TestCase):
    def test_assigns_new_ids(self):
        tracker = SimpleTracker()
        dets = [{"bbox": [0, 0, 50, 50]}, {"bbox": [500, 500, 550, 550]}]
        result = tracker.update(dets, FRAME_SHAPE)
        ids = {d["id"] for d in result}
        self.assertEqual(ids, {1, 2})

    def test_maintains_id_across_small_movement(self):
        tracker = SimpleTracker()
        dets1 = [{"bbox": [100, 100, 150, 150]}]
        tracker.update(dets1, FRAME_SHAPE)

        dets2 = [{"bbox": [105, 103, 155, 153]}]  # small move
        result = tracker.update(dets2, FRAME_SHAPE)
        self.assertEqual(result[0]["id"], 1)

    def test_new_id_for_distant_detection(self):
        tracker = SimpleTracker(max_distance_ratio=0.05)
        dets1 = [{"bbox": [0, 0, 50, 50]}]
        tracker.update(dets1, FRAME_SHAPE)

        dets2 = [{"bbox": [1000, 600, 1050, 650]}]  # far away
        result = tracker.update(dets2, FRAME_SHAPE)
        self.assertNotEqual(result[0]["id"], 1)

    def test_track_expires_after_max_lost(self):
        tracker = SimpleTracker(max_lost=2)
        tracker.update([{"bbox": [0, 0, 50, 50]}], FRAME_SHAPE)
        self.assertEqual(len(tracker.tracks), 1)

        tracker.update([], FRAME_SHAPE)  # lost=1
        tracker.update([], FRAME_SHAPE)  # lost=2
        tracker.update([], FRAME_SHAPE)  # lost=3 > max_lost -> removed
        self.assertEqual(len(tracker.tracks), 0)


if __name__ == "__main__":
    unittest.main()
