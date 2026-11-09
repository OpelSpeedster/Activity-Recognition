"""Event and fixed-window evaluation metrics for the HAR pipeline.

Event matching remains one-to-one by type and temporal overlap. Fixed-window
metrics provide a single-label confusion matrix; if multiple classes overlap
inside a window, the precedence is fall > fight > crowd_density > normal.
"""
from dataclasses import dataclass, field
from statistics import median


EVENT_TYPES = ("fall", "fight", "crowd_density")
WINDOW_LABELS = ("normal", "fall", "fight", "crowd_density")
WINDOW_EVENT_PRECEDENCE = ("fall", "fight", "crowd_density")


@dataclass
class Event:
    event_type: str
    start_frame: int
    end_frame: int
    person_id: int = None


@dataclass
class EvalResult:
    event_type: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_pred_details: list = field(default_factory=list)
    video_duration_seconds: float = 0.0

    @property
    def precision(self):
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self):
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def miss_rate(self):
        support = self.true_positives + self.false_negatives
        return self.false_negatives / support if support else 0.0

    @property
    def false_alerts_per_hour(self):
        if self.video_duration_seconds <= 0:
            return None
        return self.false_positives / (self.video_duration_seconds / 3600.0)

    @property
    def mean_temporal_iou(self):
        values = [d["temporal_iou"] for d in self.matched_pred_details]
        return sum(values) / len(values) if values else 0.0

    @property
    def average_detection_delay_seconds(self):
        return _average(self._detection_delays_seconds)

    @property
    def median_detection_delay_seconds(self):
        values = self._detection_delays_seconds
        return median(values) if values else None

    @property
    def maximum_detection_delay_seconds(self):
        values = self._detection_delays_seconds
        return max(values) if values else None

    @property
    def detected_before_event_end_percent(self):
        support = self.true_positives + self.false_negatives
        if not support:
            return 0.0
        detected_in_time = sum(
            detail["prediction"].start_frame <= detail["ground_truth"].end_frame
            for detail in self.matched_pred_details
        )
        return 100.0 * detected_in_time / support

    @property
    def _detection_delays_seconds(self):
        return [
            detail["detection_delay_seconds"]
            for detail in self.matched_pred_details
            if detail["detection_delay_seconds"] is not None
        ]

    def as_dict(self):
        return {
            "event_type": self.event_type,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "miss_rate": round(self.miss_rate, 4),
            "false_alerts_per_hour": (
                round(self.false_alerts_per_hour, 4)
                if self.false_alerts_per_hour is not None else None
            ),
            "mean_temporal_iou": round(self.mean_temporal_iou, 4),
            "average_detection_delay_seconds": _round_or_none(
                self.average_detection_delay_seconds
            ),
            "median_detection_delay_seconds": _round_or_none(
                self.median_detection_delay_seconds
            ),
            "maximum_detection_delay_seconds": _round_or_none(
                self.maximum_detection_delay_seconds
            ),
            "detected_before_event_end_percent": round(
                self.detected_before_event_end_percent, 2
            ),
        }


@dataclass
class WindowClassificationResult:
    """Single-label confusion matrix for fixed-length video windows."""

    labels: tuple
    matrix: list

    def __post_init__(self):
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Window labels must be unique")
        if len(self.matrix) != len(self.labels) or any(
            len(row) != len(self.labels) for row in self.matrix
        ):
            raise ValueError("Confusion matrix dimensions must match labels")
        if any(value < 0 for row in self.matrix for value in row):
            raise ValueError("Confusion matrix counts cannot be negative")

    @property
    def total_windows(self):
        return sum(sum(row) for row in self.matrix)

    @property
    def accuracy(self):
        if not self.total_windows:
            return 0.0
        correct = sum(self.matrix[i][i] for i in range(len(self.labels)))
        return correct / self.total_windows

    def per_class_metrics(self):
        total = self.total_windows
        results = {}
        for class_index, label in enumerate(self.labels):
            tp = self.matrix[class_index][class_index]
            support = sum(self.matrix[class_index])
            predicted = sum(row[class_index] for row in self.matrix)
            fn = support - tp
            fp = predicted - tp
            tn = total - tp - fn - fp
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            specificity_denominator = tn + fp
            specificity = tn / specificity_denominator if specificity_denominator else 0.0
            f1_denom = precision + recall
            f1 = 2 * precision * recall / f1_denom if f1_denom else 0.0
            results[label] = {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "false_positive_rate": 1.0 - specificity,
                "balanced_accuracy": (recall + specificity) / 2.0,
            }
        return results

    @property
    def macro_f1(self):
        rows = self.per_class_metrics()
        return _average([rows[label]["f1"] for label in self.labels])

    @property
    def weighted_f1(self):
        rows = self.per_class_metrics()
        if not self.total_windows:
            return 0.0
        return sum(
            rows[label]["f1"] * rows[label]["support"]
            for label in self.labels
        ) / self.total_windows

    @property
    def balanced_accuracy(self):
        """Multiclass balanced accuracy: macro-average of class recall."""
        rows = self.per_class_metrics()
        return _average([rows[label]["recall"] for label in self.labels])

    def as_dict(self):
        return {
            "labels": list(self.labels),
            "matrix": [list(row) for row in self.matrix],
            "total_windows": self.total_windows,
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "per_class": {
                label: {
                    key: (round(value, 4) if isinstance(value, float) else value)
                    for key, value in metrics.items()
                }
                for label, metrics in self.per_class_metrics().items()
            },
        }


def _average(values):
    return sum(values) / len(values) if values else None


def _round_or_none(value):
    return round(value, 4) if value is not None else None


def event_iou(a, b):
    """Temporal intersection-over-union for inclusive frame intervals."""
    intersection = max(0, min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame) + 1)
    if intersection == 0:
        return 0.0
    union = max(a.end_frame, b.end_frame) - min(a.start_frame, b.start_frame) + 1
    return intersection / union if union > 0 else 0.0


def _overlaps(a_start, a_end, b_start, b_end, tolerance_frames):
    return not (a_end + tolerance_frames < b_start or b_end + tolerance_frames < a_start)


def _same_person_if_known(ground_truth, prediction):
    return (
        ground_truth.person_id is None
        or prediction.person_id is None
        or ground_truth.person_id == prediction.person_id
    )


def evaluate_events(
    ground_truth,
    predictions,
    event_type,
    tolerance_frames=10,
    fps=None,
    video_duration_seconds=0.0,
    min_event_iou=None,
):
    """Match same-type events and compute event metrics.

    By default, events match when their intervals overlap within tolerance.
    Set ``min_event_iou`` (for example, 0.5) to require a stricter temporal
    intersection-over-union instead. A detected event's delay is its predicted
    start frame minus the ground-truth start frame; early alerts are negative.
    """
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be non-negative")
    if min_event_iou is not None and not 0 < min_event_iou <= 1:
        raise ValueError("min_event_iou must be greater than 0 and at most 1")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive")

    gt = [e for e in ground_truth if e.event_type == event_type]
    pred = [e for e in predictions if e.event_type == event_type]

    edges = []
    for p in pred:
        candidates = []
        for gi, g in enumerate(gt):
            if not _same_person_if_known(g, p):
                continue
            if min_event_iou is None:
                is_match = _overlaps(
                    p.start_frame, p.end_frame, g.start_frame, g.end_frame,
                    tolerance_frames,
                )
            else:
                is_match = event_iou(p, g) >= min_event_iou
            if is_match:
                candidates.append(gi)
        edges.append(candidates)

    matched_gt_to_pred = {}

    def augment(pred_index, seen_gt):
        for gt_index in edges[pred_index]:
            if gt_index in seen_gt:
                continue
            seen_gt.add(gt_index)
            previous_pred = matched_gt_to_pred.get(gt_index)
            if previous_pred is None or augment(previous_pred, seen_gt):
                matched_gt_to_pred[gt_index] = pred_index
                return True
        return False

    for pred_index in sorted(range(len(pred)), key=lambda i: len(edges[i])):
        augment(pred_index, set())

    matched_pred = set(matched_gt_to_pred.values())
    matched_pairs = sorted(
        ((pred_index, gt_index) for gt_index, pred_index in matched_gt_to_pred.items()),
        key=lambda pair: pair[0],
    )
    details = []
    for pred_index, gt_index in matched_pairs:
        predicted_event = pred[pred_index]
        truth_event = gt[gt_index]
        delay_frames = predicted_event.start_frame - truth_event.start_frame
        details.append({
            "prediction": predicted_event,
            "ground_truth": truth_event,
            "temporal_iou": event_iou(predicted_event, truth_event),
            "detection_delay_frames": delay_frames,
            "detection_delay_seconds": delay_frames / fps if fps else None,
        })

    return EvalResult(
        event_type=event_type,
        true_positives=len(matched_pairs),
        false_positives=len(pred) - len(matched_pred),
        false_negatives=len(gt) - len(matched_gt_to_pred),
        matched_pred_details=details,
        video_duration_seconds=max(0.0, video_duration_seconds),
    )


def summarize(results, event_type=None, video_duration_seconds=None):
    """Combine event counts and match details without weighting by duration."""
    results = list(results)
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)
    if event_type is None:
        event_types = {r.event_type for r in results}
        event_type = next(iter(event_types)) if len(event_types) == 1 else "overall"
    if video_duration_seconds is None:
        if len({r.event_type for r in results}) <= 1:
            video_duration_seconds = sum(r.video_duration_seconds for r in results)
        else:
            video_duration_seconds = max(
                (r.video_duration_seconds for r in results), default=0.0
            )
    return EvalResult(
        event_type=event_type,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        matched_pred_details=[
            detail for result in results for detail in result.matched_pred_details
        ],
        video_duration_seconds=video_duration_seconds,
    )


def _window_class(events, start_frame, end_frame, labels):
    active_types = {
        event.event_type
        for event in events
        if event.start_frame <= end_frame and event.end_frame >= start_frame
    }
    unknown = active_types - set(labels)
    if unknown:
        raise ValueError(f"Events use classes not in the window labels: {sorted(unknown)}")
    for event_type in WINDOW_EVENT_PRECEDENCE:
        if event_type in active_types:
            return event_type
    return "normal"


def confusion_matrix_windows(
    ground_truth,
    predictions,
    total_frames,
    window_frames,
    labels=WINDOW_LABELS,
):
    """Convert intervals to a single-label confusion matrix over fixed windows.

    Any event overlap marks the window positive. If more than one class is
    present, the fixed safety priority is fall > fight > crowd_density. Empty
    windows are normal. The final window may be shorter than ``window_frames``.
    """
    labels = tuple(labels)
    if "normal" not in labels:
        raise ValueError("Window labels must include 'normal'")
    if total_frames < 0:
        raise ValueError("total_frames must be non-negative")
    if window_frames < 1:
        raise ValueError("window_frames must be at least 1")

    class_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for start_frame in range(1, total_frames + 1, window_frames):
        end_frame = min(total_frames, start_frame + window_frames - 1)
        actual = _window_class(ground_truth, start_frame, end_frame, labels)
        predicted = _window_class(predictions, start_frame, end_frame, labels)
        matrix[class_index[actual]][class_index[predicted]] += 1
    return WindowClassificationResult(labels, matrix)


def combine_window_results(results):
    """Sum fixed-window confusion counts from multiple videos."""
    results = list(results)
    if not results:
        return WindowClassificationResult(WINDOW_LABELS, [[0] * len(WINDOW_LABELS) for _ in WINDOW_LABELS])
    labels = results[0].labels
    if any(result.labels != labels for result in results):
        raise ValueError("Cannot combine confusion matrices with different labels")
    size = len(labels)
    matrix = [[0] * size for _ in range(size)]
    for result in results:
        for row in range(size):
            for column in range(size):
                matrix[row][column] += result.matrix[row][column]
    return WindowClassificationResult(labels, matrix)


def print_report(video_name, results, overall, matching_description=None):
    print(f"\n=== Event evaluation: {video_name} ===")
    if matching_description:
        print(f"Event matching: {matching_description}")
    print("class          TP   FP   FN   Precision Recall F1    Miss% IoU   Avg/Med/Max delay(s)    Before-end% FP/hour")
    for result in results:
        fp_per_hour = (
            f"{result.false_alerts_per_hour:.2f}"
            if result.false_alerts_per_hour is not None else "n/a"
        )
        avg_delay = result.average_detection_delay_seconds
        med_delay = result.median_detection_delay_seconds
        max_delay = result.maximum_detection_delay_seconds
        delays = (
            f"{avg_delay:.2f}/{med_delay:.2f}/{max_delay:.2f}"
            if avg_delay is not None and med_delay is not None and max_delay is not None
            else "n/a"
        )
        print(
            f"{result.event_type:14s} {result.true_positives:3d} {result.false_positives:4d} "
            f"{result.false_negatives:4d} {result.precision:9.3f} {result.recall:6.3f} "
            f"{result.f1:5.3f} {100 * result.miss_rate:5.1f} {result.mean_temporal_iou:5.3f} "
            f"{delays:23s} {result.detected_before_event_end_percent:10.1f} {fp_per_hour:>8s}"
        )

    supports = [r.true_positives + r.false_negatives for r in results]
    if results:
        macro_f1 = _average([r.f1 for r in results]) or 0.0
        total_support = sum(supports)
        weighted_f1 = (
            sum(r.f1 * support for r, support in zip(results, supports)) / total_support
            if total_support else 0.0
        )
    else:
        macro_f1 = weighted_f1 = 0.0
    overall_fp_rate = (
        f"{overall.false_alerts_per_hour:.2f}"
        if overall.false_alerts_per_hour is not None else "n/a"
    )
    print(
        f"Event macro-F1={macro_f1:.3f}  weighted-F1={weighted_f1:.3f}  "
        f"micro-F1={overall.f1:.3f}  miss-rate={overall.miss_rate:.3f}  "
        f"overall FP/hour={overall_fp_rate}"
    )
    print(
        "PR curve/AP: unavailable because the current rule detectors emit binary "
        "events, not confidence scores."
    )


def print_window_report(video_name, result, window_seconds):
    print(f"\n=== {window_seconds:g}s window metrics: {video_name} ===")
    print("Confusion matrix (actual rows, predicted columns):")
    print("actual\\pred  " + "  ".join(f"{label:>13s}" for label in result.labels))
    for label, row in zip(result.labels, result.matrix):
        print(f"{label:12s}  " + "  ".join(f"{value:13d}" for value in row))
    print(
        f"Accuracy={result.accuracy:.3f}  Balanced accuracy={result.balanced_accuracy:.3f}  "
        f"Macro-F1={result.macro_f1:.3f}  Weighted-F1={result.weighted_f1:.3f}  "
        f"Windows={result.total_windows}"
    )
    print("class          Support Precision Recall F1    Specificity FPR   OvR-BAcc")
    for label, metrics in result.per_class_metrics().items():
        print(
            f"{label:14s} {metrics['support']:7d} {metrics['precision']:9.3f} "
            f"{metrics['recall']:6.3f} {metrics['f1']:5.3f} "
            f"{metrics['specificity']:11.3f} {metrics['false_positive_rate']:5.3f} "
            f"{metrics['balanced_accuracy']:9.3f}"
        )


def accumulate_events(predictions, open_events, event_type, detections, frame_id,
                      max_gap_frames=2):
    """Group repeated per-frame hits into episodes, closing after a short gap.

    Detector records contain a person ID. A hit for that ID updates its
    episode; if it is absent for more than max_gap_frames, the episode closes
    at its last hit. This prevents an early detection from spanning the rest
    of the video and masking later false positives or missed events.
    """
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")

    active_ids = {
        detection.get("id")
        for detection in detections
        if detection.get("id") is not None
    }

    for key, state in list(open_events.items()):
        if key[0] == event_type and key[1] not in active_ids:
            if frame_id - state["last_frame"] > max_gap_frames:
                predictions.append(Event(key[0], state["start_frame"], state["last_frame"], key[1]))
                del open_events[key]

    for person_id in sorted(active_ids, key=str):
        key = (event_type, person_id)
        state = open_events.get(key)
        if state is None:
            open_events[key] = {"start_frame": frame_id, "last_frame": frame_id}
        else:
            state["last_frame"] = frame_id


def finalize_open_events(predictions, open_events):
    """Close remaining episodes at their actual last detection frame."""
    for (event_type, person_id), state in list(open_events.items()):
        predictions.append(
            Event(event_type, state["start_frame"], state["last_frame"], person_id)
        )
        del open_events[(event_type, person_id)]
