"""
Run the real detection pipeline (pose estimation -> tracking -> fall/fight
detectors) over a labeled test video and report precision/recall/F1
against ground truth.

Usage:
    python -m eval.run_eval --video path/to/clip.mp4 --labels eval/labels/clip.csv
    python -m eval.run_eval --video-dir eval/clips --labels-dir eval/labels

Requires `ultralytics` + a pose model (real YOLO inference) to test the
full pipeline end-to-end. If you just want to validate the *detector logic*
without running YOLO (e.g. in an environment without a GPU or webcam), use
eval/run_eval_synthetic.py instead, which feeds synthetic keypoint
sequences directly into FallDetector/FightDetector.
"""
import argparse
import csv
from pathlib import Path

import cv2

from config import (
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
from detectors.fall_detector import FallDetector
from detectors.fight_detector import FightDetector
from detectors.pose_estimator import PoseEstimator
from eval.metrics import Event, evaluate_events, summarize, print_report, accumulate_events
from utils.simple_tracker import SimpleTracker
from utils.model_profile import resolve_pose_profile


def load_ground_truth(labels_path):
    events = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(
                Event(
                    event_type=row["event_type"].strip(),
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    person_id=int(row["person_id"]) if row.get("person_id") else None,
                )
            )
    return events


def run_pipeline_on_video(video_path, device=None):
    """Runs the real pose->track->detect pipeline, returns predicted Events."""
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
    tracker = SimpleTracker(max_lost=TRACK_MAX_LOST, max_distance_ratio=TRACK_MAX_DISTANCE_RATIO)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    predictions = []
    open_events = {}  # (event_type, person_id) -> start_frame, for merging consecutive detections
    frame_id = 0

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

        accumulate_events(predictions, open_events, "fall", falls, frame_id)
        accumulate_events(predictions, open_events, "fight", fights, frame_id)

    # close any still-open events at end of video
    for (etype, pid), start in open_events.items():
        predictions.append(Event(etype, start, frame_id, pid))

    cap.release()
    return predictions


def evaluate_video(video_path, labels_path, device=None):
    ground_truth = load_ground_truth(labels_path)
    predictions = run_pipeline_on_video(video_path, device=device)

    results = []
    for event_type in ("fall", "fight"):
        results.append(evaluate_events(ground_truth, predictions, event_type))

    overall = summarize(results)
    return results, overall, predictions, ground_truth


def main():
    parser = argparse.ArgumentParser(description="Evaluate HAR pipeline against labeled video(s)")
    parser.add_argument("--video", help="Single video file")
    parser.add_argument("--labels", help="Matching ground-truth CSV for --video")
    parser.add_argument("--video-dir", help="Directory of videos, matched by filename stem to --labels-dir")
    parser.add_argument("--labels-dir", help="Directory of label CSVs")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    all_results = []
    if args.video and args.labels:
        results, overall, _, _ = evaluate_video(args.video, args.labels, device=args.device)
        print_report(Path(args.video).name, results, overall)
        all_results.extend(results)
    elif args.video_dir and args.labels_dir:
        video_dir = Path(args.video_dir)
        labels_dir = Path(args.labels_dir)
        for video_path in sorted(video_dir.glob("*.mp4")):
            label_path = labels_dir / f"{video_path.stem}.csv"
            if not label_path.exists():
                print(f"[skip] no labels for {video_path.name}")
                continue
            results, overall, _, _ = evaluate_video(video_path, label_path, device=args.device)
            print_report(video_path.name, results, overall)
            all_results.extend(results)
    else:
        parser.error("Provide either --video/--labels or --video-dir/--labels-dir")

    if len(all_results) > 2:
        fall_results = [r for r in all_results if r.event_type == "fall"]
        fight_results = [r for r in all_results if r.event_type == "fight"]
        print("\n=== Aggregate across all videos ===")
        print_report("ALL", [summarize(fall_results), summarize(fight_results)], summarize(all_results))


if __name__ == "__main__":
    main()
