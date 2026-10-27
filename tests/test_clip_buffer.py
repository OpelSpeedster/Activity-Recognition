import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from utils.clip_buffer import ClipBuffer, sample_event_frames


def marker(value):
    return np.full((2, 2, 3), value, dtype=np.uint8)


class TestSampleEventFrames(unittest.TestCase):
    def test_selects_pre_trigger_and_post_trigger_frames(self):
        frames = [marker(i) for i in range(9)]
        samples = sample_event_frames(frames, trigger_index=4, pre_count=4, post_count=4)
        self.assertEqual([int(frame[0, 0, 0]) for frame in samples], [0, 4, 8])

    def test_rejects_missing_temporal_phase(self):
        with self.assertRaises(ValueError):
            sample_event_frames([marker(0), marker(1)], 1, pre_count=1, post_count=1)


class TestClipBuffer(unittest.TestCase):
    def test_clip_contains_pre_event_trigger_and_post_event_frames(self):
        with TemporaryDirectory() as directory:
            buffer = ClipBuffer(fps=2, seconds=4, output_dir=Path(directory))
            with patch.object(ClipBuffer, "_write_event_clip", return_value="event.mp4"):
                # Older frames outside the configured pre-event half must not
                # lengthen the final clip.
                for value in range(9):
                    buffer.add(marker(value))
                self.assertTrue(buffer.trigger("event-1", metadata={"frame_id": 5}))
                completed = []
                for value in range(9, 13):
                    completed.extend(buffer.add(marker(value)))

            self.assertEqual(len(completed), 1)
            record = completed[0]
            self.assertEqual(record["trigger_index"], 4)
            self.assertEqual(record["metadata"], {"frame_id": 5})
            self.assertEqual([int(frame[0, 0, 0]) for frame in record["frames"]], list(range(4, 13)))

    def test_flush_pending_writes_short_clip_at_end_of_input(self):
        with TemporaryDirectory() as directory:
            buffer = ClipBuffer(fps=2, seconds=4, output_dir=Path(directory))
            with patch.object(ClipBuffer, "_write_event_clip", return_value="short.mp4"):
                for value in range(5):
                    buffer.add(marker(value))
                buffer.trigger("event-1", metadata={})
                buffer.add(marker(5))
                completed = buffer.flush_pending()

            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]["clip_path"], "short.mp4")
            self.assertEqual(len(completed[0]["frames"]), 6)


if __name__ == "__main__":
    unittest.main()