import unittest
from contextlib import redirect_stdout
from io import StringIO

from eval.metrics import (
    Event,
    WindowClassificationResult,
    combine_window_results,
    confusion_matrix_windows,
    evaluate_events,
    event_iou,
    print_window_report,
    summarize,
)


class TestEventMetrics(unittest.TestCase):
    def test_temporal_iou_uses_inclusive_frame_intervals(self):
        self.assertAlmostEqual(
            event_iou(Event('fall', 100, 180), Event('fall', 120, 190)),
            61 / 91,
        )

    def test_event_report_tracks_delay_iou_miss_rate_and_false_alert_rate(self):
        truth = [Event('fall', 100, 120)]
        predictions = [Event('fall', 110, 130), Event('fall', 300, 310)]
        result = evaluate_events(
            truth, predictions, 'fall', fps=25, video_duration_seconds=3600
        )

        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (1, 1, 0))
        self.assertEqual(result.miss_rate, 0)
        self.assertEqual(result.false_alerts_per_hour, 1)
        self.assertAlmostEqual(result.average_detection_delay_seconds, 0.4)
        self.assertEqual(result.detected_before_event_end_percent, 100)
        self.assertAlmostEqual(result.mean_temporal_iou, 11 / 31)

    def test_iou_threshold_can_make_matching_stricter(self):
        truth = [Event('fall', 100, 180)]
        predictions = [Event('fall', 120, 190)]
        result = evaluate_events(truth, predictions, 'fall', min_event_iou=0.7)
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (0, 1, 1))

    def test_summary_preserves_matched_details_and_explicit_video_duration(self):
        result = evaluate_events(
            [Event('fall', 10, 20)], [Event('fall', 12, 15)], 'fall',
            fps=10, video_duration_seconds=5,
        )
        summary = summarize([result], video_duration_seconds=10)
        self.assertEqual(summary.true_positives, 1)
        self.assertEqual(summary.video_duration_seconds, 10)
        self.assertEqual(len(summary.matched_pred_details), 1)


class TestWindowClassificationMetrics(unittest.TestCase):
    def test_confusion_matrix_classifies_normal_fall_and_fight_windows(self):
        truth = [Event('fall', 31, 60), Event('fight', 61, 90)]
        predictions = [Event('fight', 31, 60)]
        result = confusion_matrix_windows(
            truth, predictions, total_frames=90, window_frames=30,
            labels=('normal', 'fall', 'fight'),
        )
        self.assertEqual(
            result.matrix,
            [[1, 0, 0], [0, 0, 1], [1, 0, 0]],
        )
        self.assertAlmostEqual(result.accuracy, 1 / 3)

    def test_last_partial_window_is_included(self):
        result = confusion_matrix_windows([], [], total_frames=65, window_frames=30)
        self.assertEqual(result.total_windows, 3)
        self.assertEqual(result.matrix[0][0], 3)

    def test_specificity_balanced_accuracy_macro_and_weighted_f1(self):
        metrics = WindowClassificationResult(
            labels=('normal', 'fall', 'fight'),
            matrix=[[8, 1, 1], [2, 4, 0], [1, 1, 5]],
        )
        per_class = metrics.per_class_metrics()
        self.assertAlmostEqual(metrics.accuracy, 17 / 23)
        self.assertAlmostEqual(per_class['normal']['specificity'], 10 / 13)
        self.assertAlmostEqual(per_class['fall']['false_positive_rate'], 2 / 17)
        self.assertAlmostEqual(metrics.balanced_accuracy, sum(per_class[k]['recall'] for k in metrics.labels) / 3)
        self.assertAlmostEqual(metrics.macro_f1, sum(per_class[k]['f1'] for k in metrics.labels) / 3)
        self.assertAlmostEqual(
            metrics.weighted_f1,
            sum(per_class[k]['f1'] * per_class[k]['support'] for k in metrics.labels) / 23,
        )

    def test_event_priority_is_deterministic_for_overlapping_labels(self):
        result = confusion_matrix_windows(
            [Event('fight', 1, 30), Event('fall', 1, 30)], [],
            total_frames=30, window_frames=30,
        )
        self.assertEqual(result.matrix[1][0], 1)  # fall is the first actual class after normal

    def test_confusion_matrices_combine_counts(self):
        first = confusion_matrix_windows([], [], total_frames=30, window_frames=30)
        second = confusion_matrix_windows([], [], total_frames=30, window_frames=30)
        combined = combine_window_results([first, second])
        self.assertEqual(combined.total_windows, 2)
        self.assertEqual(combined.matrix[0][0], 2)

    def test_window_report_accepts_seconds_argument(self):
        metrics = confusion_matrix_windows([], [], total_frames=30, window_frames=30)
        output = StringIO()
        with redirect_stdout(output):
            print_window_report('clip.mp4', metrics, 1.0)
        self.assertIn('1s window metrics', output.getvalue())


if __name__ == '__main__':
    unittest.main()
