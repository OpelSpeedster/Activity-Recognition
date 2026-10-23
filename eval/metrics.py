"""
Event-level evaluation metrics for the HAR pipeline.

Why not simple per-frame accuracy?
-----------------------------------
Naive per-frame accuracy is misleading here: if "no fall" is true for 95%
of frames in a video, a detector that NEVER predicts a fall still scores
95% frame accuracy while being completely useless. This is the classic
class-imbalance trap.

Instead we evaluate at the EVENT level, which is what actually matters for
a safety system ("did it correctly catch the fall that happened, and did
it avoid crying wolf"):

  - A ground-truth event is COUNTED AS DETECTED (true positive) if the
    system fired a matching-type prediction with any overlap +/- a
    tolerance window around [start_frame, end_frame].
  - A ground-truth event with no matching prediction is a FALSE NEGATIVE
    (a real fall/fight the system missed — the costliest failure mode for
    a safety system).
  - A predicted event with no overlapping ground-truth event is a FALSE
    POSITIVE (a false alert — costly in a different way because it can
    create alert fatigue for the person reviewing events).

Metrics reported: Precision, Recall, F1, and a confusion-style summary,
per event type (fall / fight) and overall.
"""
from dataclasses import dataclass, field


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

    def as_dict(self):
        return {
            "event_type": self.event_type,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def _overlaps(a_start, a_end, b_start, b_end, tolerance_frames):
    return not (a_end + tolerance_frames < b_start or b_end + tolerance_frames < a_start)


def evaluate_events(ground_truth, predictions, event_type, tolerance_frames=10):
    """
    ground_truth / predictions: lists of Event with .event_type matching event_type.
    tolerance_frames: how many frames of slack allowed around the labeled window,
        to account for reasonable detection latency (debounce frames, etc.)
        without being so loose that it hides real misses.

    Matching is greedy one-to-one: each ground-truth event can be matched by
    at most one prediction and vice versa, so duplicate predictions for the
    same real event don't inflate true positives.
    """
    gt = [e for e in ground_truth if e.event_type == event_type]
    pred = [e for e in predictions if e.event_type == event_type]

    result = EvalResult(event_type=event_type)
    matched_gt = set()
    matched_pred = set()

    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            if gi in matched_gt:
                continue
            if _overlaps(p.start_frame, p.end_frame, g.start_frame, g.end_frame, tolerance_frames):
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    result.true_positives = len(matched_pred)
    result.false_positives = len(pred) - len(matched_pred)
    result.false_negatives = len(gt) - len(matched_gt)
    return result


def summarize(results):
    """Combine per-type EvalResults into an overall micro-averaged summary."""
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)

    overall = EvalResult(event_type="overall", true_positives=total_tp,
                          false_positives=total_fp, false_negatives=total_fn)
    return overall


def print_report(video_name, results, overall):
    print(f"\n=== Evaluation report: {video_name} ===")
    for r in results:
        d = r.as_dict()
        print(
            f"  {d['event_type']:8s}  TP={d['true_positives']:<3d} "
            f"FP={d['false_positives']:<3d} FN={d['false_negatives']:<3d}  "
            f"P={d['precision']:.2f}  R={d['recall']:.2f}  F1={d['f1']:.2f}"
        )
    o = overall.as_dict()
    print(
        f"  {'OVERALL':8s}  TP={o['true_positives']:<3d} "
        f"FP={o['false_positives']:<3d} FN={o['false_negatives']:<3d}  "
        f"P={o['precision']:.2f}  R={o['recall']:.2f}  F1={o['f1']:.2f}"
    )


def accumulate_events(predictions, open_events, event_type, detections, frame_id):
    """Merge per-frame detector hits for the same person into single events,
    so N consecutive fall-frames become ONE fall event, not N.
    Simplified: events close at end-of-video (see callers), not on gaps."""
    active_ids = {d["id"] for d in detections}
    for pid in active_ids:
        key = (event_type, pid)
        if key not in open_events:
            open_events[key] = frame_id
