"""
Run the real detection pipeline (pose estimation -> tracking -> fall/fight/
crowd-density detectors) over labeled test videos and report precision/recall/F1
against ground truth.

Usage:
    python -m eval.run_eval --video path/to/clip.mp4 --labels eval/labels/clip.csv
    python -m eval.run_eval --video-dir test_video --labels-dir eval/labels
    python -m eval.run_eval --video-dir test_video --labels-dir eval/labels --window-seconds 2
    python -m eval.run_eval --video path/to/clip.mp4 --labels eval/labels/clip.csv --min-event-iou 0.5

Requires `ultralytics` + a pose model (real YOLO inference) to test the
full pipeline end-to-end. If you just want to validate the *detector logic*
without running YOLO (e.g. in an environment without a GPU or webcam), use
eval/run_eval_synthetic.py instead, which feeds synthetic keypoint
sequences directly into FallDetector/FightDetector.
"""
import argparse
import csv
import math
from pathlib import Path

import cv2

from config import (
    DENSITY_AREA_SCALE,
    DENSITY_HIGH,
    DENSITY_MEDIUM,
    FALL_ANGLE_THRESHOLD,
    FALL_MIN_FRAMES,
    FALL_MIN_DROP_RATIO,
    FIGHT_MIN_FRAMES,
    FIGHT_MOTION_THRESHOLD,
    FIGHT_NEAR_DISTANCE_RATIO,
    FIGHT_SPIKE_THRESHOLD,
    HORIZONTAL_RATIO_THRESHOLD,
    INACTIVITY_FRAMES,
    INACTIVITY_MOTION_THRESHOLD,
    KEYPOINT_CONF_THRESHOLD,
    MOTION_HISTORY,
    TRACK_MAX_DISTANCE_RATIO,
    TRACK_MAX_LOST,
    YOLO_POSE_CONF,
    YOLO_POSE_IOU,
    YOLO_POSE_MAX_DET,
)
from detectors.density_estimator import DensityEstimator
from detectors.fall_detector import FallDetector
from detectors.fight_detector import FightDetector
from eval.metrics import (
    EVENT_TYPES,
    Event,
    accumulate_events,
    combine_window_results,
    confusion_matrix_windows,
    evaluate_events,
    finalize_open_events,
    print_report,
    print_window_report,
    summarize,
)
from utils.simple_tracker import SimpleTracker
from utils.model_profile import resolve_pose_profile


def load_ground_truth(labels_path):
    events = []
    required = {"event_type", "start_frame", "end_frame", "person_id"}
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Label CSV is missing columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            event_type = str(row.get("event_type") or "").strip()
            if event_type not in {"fall", "fight", "crowd_density"}:
                raise ValueError(
                    f"{labels_path}:{row_number}: event_type must be 'fall', 'fight', or 'crowd_density'"
                )
            try:
                start_frame = int(row["start_frame"])
                end_frame = int(row["end_frame"])
                person_id_text = str(row.get("person_id") or "").strip()
                person_id = int(person_id_text) if person_id_text else None
            except ValueError as exc:
                raise ValueError(f"{labels_path}:{row_number}: frame and person IDs must be integers") from exc
            if start_frame < 1 or end_frame < start_frame:
                raise ValueError(
                    f"{labels_path}:{row_number}: expected 1 <= start_frame <= end_frame"
                )
            events.append(Event(event_type, start_frame, end_frame, person_id))
    return events


def run_pipeline_on_video(video_path, device=None):
    """Runs the real pose->track->detect pipeline, returns predicted Events."""
    from detectors.pose_estimator import PoseEstimator

    model_path, resolved_device, image_size = resolve_pose_profile(device)
    pose_estimator = PoseEstimator(
        model_path,
        conf=YOLO_POSE_CONF,
        iou=YOLO_POSE_IOU,
        device=resolved_device,
        imgsz=image_size,
        max_det=YOLO_POSE_MAX_DET,
    )
    fall_detector = FallDetector(
        angle_threshold=FALL_ANGLE_THRESHOLD,
        min_frames=FALL_MIN_FRAMES,
        inactivity_frames=INACTIVITY_FRAMES,
        inactivity_motion_threshold=INACTIVITY_MOTION_THRESHOLD,
        horizontal_ratio_threshold=HORIZONTAL_RATIO_THRESHOLD,
        min_drop_ratio=FALL_MIN_DROP_RATIO,
        keypoint_conf_threshold=KEYPOINT_CONF_THRESHOLD,
    )
    fight_detector = FightDetector(
        motion_threshold=FIGHT_MOTION_THRESHOLD,
        spike_threshold=FIGHT_SPIKE_THRESHOLD,
        near_distance_ratio=FIGHT_NEAR_DISTANCE_RATIO,
        history_len=MOTION_HISTORY,
        min_consec_frames=FIGHT_MIN_FRAMES,
        keypoint_conf_threshold=KEYPOINT_CONF_THRESHOLD,
    )
    density_estimator = DensityEstimator(
        area_scale=DENSITY_AREA_SCALE,
        medium_threshold=DENSITY_MEDIUM,
        high_threshold=DENSITY_HIGH,
    )
    tracker = SimpleTracker(max_lost=TRACK_MAX_LOST, max_distance_ratio=TRACK_MAX_DISTANCE_RATIO)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    predictions = []
    open_events = {}  # (event_type, person_id) -> episode start and last-hit frames
    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            detections, keypoints_list = pose_estimator.estimate(frame)
            detections = tracker.update(detections, frame.shape)
            poses = [
                {"id": d.get("id", -1), "bbox": d["bbox"], "keypoints": kp}
                for d, kp in zip(detections, keypoints_list)
            ]

            falls, _inactivity = fall_detector.update(poses)
            fights = fight_detector.update(poses, frame.shape)
            density = density_estimator.estimate(detections, frame.shape)

            accumulate_events(predictions, open_events, "fall", falls, frame_id)
            accumulate_events(predictions, open_events, "fight", fights, frame_id)
            density_hits = (
                [{"id": -1}]
                if density["level"] in {"medium", "high"}
                else []
            )
            accumulate_events(
                predictions, open_events, "crowd_density", density_hits, frame_id
            )
    finally:
        cap.release()

    finalize_open_events(predictions, open_events)
    return predictions


def get_video_metadata(video_path):
    """Return the video's frame rate, frame count, and duration in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has an invalid frame rate: {video_path}")
        if frame_count <= 0:
            frame_count = 0
            while True:
                ok, _frame = cap.read()
                if not ok:
                    break
                frame_count += 1
        if frame_count <= 0:
            raise ValueError(f"Video contains no readable frames: {video_path}")
        return fps, frame_count, frame_count / fps
    finally:
        cap.release()


def evaluate_video(
    video_path,
    labels_path,
    device=None,
    tolerance_frames=10,
    window_seconds=1.0,
    min_event_iou=None,
):
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    ground_truth = load_ground_truth(labels_path)
    fps, frame_count, duration_seconds = get_video_metadata(video_path)
    predictions = run_pipeline_on_video(video_path, device=device)
    results = [
        evaluate_events(
            ground_truth,
            predictions,
            event_type,
            tolerance_frames=tolerance_frames,
            fps=fps,
            video_duration_seconds=duration_seconds,
            min_event_iou=min_event_iou,
        )
        for event_type in EVENT_TYPES
    ]
    overall = summarize(results, video_duration_seconds=duration_seconds)
    window_frames = max(1, round(fps * window_seconds))
    window_metrics = confusion_matrix_windows(
        ground_truth,
        predictions,
        total_frames=frame_count,
        window_frames=window_frames,
    )
    return results, overall, predictions, ground_truth, window_metrics, fps, duration_seconds


def main():
    parser = argparse.ArgumentParser(description="Evaluate HAR pipeline against labeled video(s)")
    parser.add_argument("--video", help="Single video file")
    parser.add_argument("--labels", help="Matching ground-truth CSV for --video")
    parser.add_argument("--video-dir", help="Directory of videos, matched by filename stem to --labels-dir")
    parser.add_argument("--labels-dir", help="Directory of label CSVs")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--tolerance-frames", type=int, default=10,
        help="Allowed event-boundary timing difference in frames (default: 10)",
    )
    parser.add_argument(
        "--window-seconds", type=float, default=1.0,
        help="Window length for the confusion matrix in seconds (default: 1)",
    )
    parser.add_argument(
        "--min-event-iou", type=float, default=None,
        help="Optional strict event match threshold, from greater than 0 to 1; e.g. 0.5",
    )
    args = parser.parse_args()
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")
    if args.min_event_iou is not None and not 0 < args.min_event_iou <= 1:
        parser.error("--min-event-iou must be greater than 0 and at most 1")
    if args.tolerance_frames < 0:
        parser.error("--tolerance-frames must be non-negative")

    all_results = []
    all_window_metrics = []
    total_video_seconds = 0.0
    if args.video and args.labels:
        evaluated = evaluate_video(
            args.video,
            args.labels,
            device=args.device,
            tolerance_frames=args.tolerance_frames,
            window_seconds=args.window_seconds,
            min_event_iou=args.min_event_iou,
        )
        results, overall, _predictions, _ground_truth, windows, fps, duration = evaluated
        matching_description = (
            f"temporal IoU >= {args.min_event_iou:.2f}"
            if args.min_event_iou is not None
            else f"overlap with {args.tolerance_frames}-frame boundary tolerance"
        )
        print_report(Path(args.video).name, results, overall, matching_description)
        print_window_report(Path(args.video).name, windows, args.window_seconds)
        all_results.extend(results)
        all_window_metrics.append(windows)
        total_video_seconds += duration
    elif args.video_dir and args.labels_dir:
        video_dir = Path(args.video_dir)
        labels_dir = Path(args.labels_dir)
        for video_path in sorted(video_dir.glob("*.mp4")):
            label_path = labels_dir / f"{video_path.stem}.csv"
            if not label_path.exists():
                print(f"[skip] no labels for {video_path.name}")
                continue
            evaluated = evaluate_video(
                video_path,
                label_path,
                device=args.device,
                tolerance_frames=args.tolerance_frames,
                window_seconds=args.window_seconds,
                min_event_iou=args.min_event_iou,
            )
            results, overall, _predictions, _ground_truth, windows, _fps, duration = evaluated
            matching_description = (
                f"temporal IoU >= {args.min_event_iou:.2f}"
                if args.min_event_iou is not None
                else f"overlap with {args.tolerance_frames}-frame boundary tolerance"
            )
            print_report(video_path.name, results, overall, matching_description)
            print_window_report(video_path.name, windows, args.window_seconds)
            all_results.extend(results)
            all_window_metrics.append(windows)
            total_video_seconds += duration
    else:
        parser.error("Provide either --video/--labels or --video-dir/--labels-dir")

    if len(all_window_metrics) > 1:
        event_results = {
            event_type: [r for r in all_results if r.event_type == event_type]
            for event_type in EVENT_TYPES
        }
        print("\n=== Aggregate across all videos ===")
        matching_description = (
            f"temporal IoU >= {args.min_event_iou:.2f}"
            if args.min_event_iou is not None
            else f"overlap with {args.tolerance_frames}-frame boundary tolerance"
        )
        print_report(
            "ALL",
            [
                summarize(
                    event_results[event_type],
                    event_type,
                    video_duration_seconds=total_video_seconds,
                )
                for event_type in EVENT_TYPES
            ],
            summarize(
                all_results,
                "overall",
                video_duration_seconds=total_video_seconds,
            ),
            matching_description,
        )
        print_window_report(
            "ALL",
            combine_window_results(all_window_metrics),
            args.window_seconds,
        )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
