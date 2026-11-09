import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.metrics import Event
from eval.run_eval import evaluate_video, load_ground_truth


class TestRealVideoEvaluatorWiring(unittest.TestCase):
    def _write_labels(self, path, rows):
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['event_type', 'start_frame', 'end_frame', 'person_id'])
            writer.writerows(rows)

    def test_load_ground_truth_accepts_all_scored_event_types(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
            path = Path(temp_dir) / 'labels.csv'
            self._write_labels(path, [
                ['fall', 1, 10, ''],
                ['fight', 11, 20, 3],
                ['crowd_density', 21, 30, ''],
            ])
            labels = load_ground_truth(path)
        self.assertEqual([event.event_type for event in labels], ['fall', 'fight', 'crowd_density'])
        self.assertIsNone(labels[0].person_id)
        self.assertEqual(labels[1].person_id, 3)

    def test_evaluate_video_returns_event_and_window_metrics(self):
        truth = [
            Event('fall', 31, 60),
            Event('crowd_density', 61, 90),
        ]
        predictions = [
            Event('fall', 31, 60),
            Event('crowd_density', 61, 90, person_id=-1),
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
            path = Path(temp_dir) / 'labels.csv'
            self._write_labels(path, [
                ['fall', 31, 60, ''],
                ['crowd_density', 61, 90, ''],
            ])
            with patch('eval.run_eval.get_video_metadata', return_value=(30.0, 90, 3.0)):
                with patch('eval.run_eval.run_pipeline_on_video', return_value=predictions):
                    results, overall, _predictions, labels, windows, fps, duration = evaluate_video(
                        'clip.mp4', path, window_seconds=1.0
                    )

        per_class = {result.event_type: result for result in results}
        self.assertEqual(per_class['fall'].true_positives, 1)
        self.assertEqual(per_class['crowd_density'].true_positives, 1)
        self.assertEqual(overall.true_positives, 2)
        self.assertEqual(labels, truth)
        self.assertEqual(windows.matrix[0][0], 1)
        self.assertEqual(windows.matrix[1][1], 1)
        self.assertEqual(windows.matrix[3][3], 1)
        self.assertEqual(fps, 30.0)
        self.assertEqual(duration, 3.0)


if __name__ == '__main__':
    unittest.main()
