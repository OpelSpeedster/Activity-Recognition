"""
Entry point: wires video input -> pose estimation -> tracking -> detectors
-> decision agent -> action executor into a real-time loop.

Run:
    python main.py --source webcam --agent rule
    python main.py --source path/to/video.mp4 --agent llm
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2

from agents.crowd_agent import CrowdSafetyAgent
from agents.rule_agent import RuleBasedAgent
from config import (
	ACTION_COOLDOWN_SECONDS,
	AGENT_CALL_EVERY_N_FRAMES,
	AGENT_MODEL,
	AGENT_TEMPERATURE,
	CLIP_DIR,
	CLIP_SECONDS,
	CAMERA_FPS,
	DISPLAY_WINDOW,
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
	FRAME_DIR,
	HORIZONTAL_RATIO_THRESHOLD,
	INACTIVITY_FRAMES,
	INACTIVITY_MOTION_THRESHOLD,
	KEYPOINT_CONF_THRESHOLD,
	LOG_DIR,
	MOTION_HISTORY,
	OPENAI_API_KEY,
	RECENT_EVENT_HISTORY,
	RISK_HIGH_STREAK_FRAMES,
	SAVE_CLIP_ON_HIGH_RISK,
	SAVE_SCREENSHOT_ON_HIGH_RISK,
	TRACK_MAX_DISTANCE_RATIO,
	TRACK_MAX_LOST,
	YOLO_POSE_CONF,
	YOLO_POSE_IOU,
	YOLO_POSE_MAX_DET,
)
from detectors.density_estimator import DensityEstimator
from detectors.fall_detector import FallDetector
from detectors.fight_detector import FightDetector
from detectors.pose_estimator import PoseEstimator
from utils.action_executor import ActionExecutor
from utils.clip_buffer import ClipBuffer
from utils.draw_utils import draw_boxes, draw_poses, draw_status
from utils.event_logger import EventLogger
from utils.simple_tracker import SimpleTracker
from utils.video_input import VideoSource
from utils.model_profile import resolve_pose_profile


def parse_args():
	parser = argparse.ArgumentParser(description="Real-time AI crowd safety system")
	parser.add_argument("--source", default="webcam", help="Use 'webcam' or provide a video file path")
	parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
	parser.add_argument("--agent", choices=["llm", "rule"], default="rule", help="Decision agent to use")
	parser.add_argument("--device", default=None, help="YOLO device like 'cpu' or 'cuda'")
	parser.add_argument("--model", default=None, help="Override the automatic YOLO pose model")
	parser.add_argument("--imgsz", type=int, default=None, help="Override inference image size")
	parser.add_argument("--no-display", action="store_true", help="Run without UI window")
	return parser.parse_args()


def analyze_with_fallback(agent, fallback_agent, context):
	try:
		return agent.analyze(context)
	except Exception as exc:
		print(f"[WARN] Agent failed, using rule-based fallback: {exc}")
		return fallback_agent.analyze(context)


def build_pipeline(args, model_path, device, image_size):
	"""Constructs all stateful pipeline components from config + CLI args."""
	pose_estimator = PoseEstimator(
		model_path,
		conf=YOLO_POSE_CONF,
		iou=YOLO_POSE_IOU,
		device=device,
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
		area_scale=DENSITY_AREA_SCALE, medium_threshold=DENSITY_MEDIUM, high_threshold=DENSITY_HIGH
	)
	tracker = SimpleTracker(max_lost=TRACK_MAX_LOST, max_distance_ratio=TRACK_MAX_DISTANCE_RATIO)
	return pose_estimator, fall_detector, fight_detector, density_estimator, tracker


def process_frame(frame, pipeline, clip_buffer):
	"""Runs one frame through the full detection stack. Pure function of
	(frame, stateful detectors) -> (poses, events, density) for testability."""
	pose_estimator, fall_detector, fight_detector, density_estimator, tracker = pipeline

	if clip_buffer:
		clip_buffer.add(frame)

	detections, keypoints_list = pose_estimator.estimate(frame)
	detections = tracker.update(detections, frame.shape)
	poses = [
		{"id": d.get("id", -1), "bbox": d["bbox"], "keypoints": kp}
		for d, kp in zip(detections, keypoints_list)
	]

	falls, inactivity = fall_detector.update(poses)
	fights = fight_detector.update(poses, frame.shape)
	density = density_estimator.estimate(detections, frame.shape)

	events = []
	for fall in falls:
		events.append({"type": "fall", "data": fall})
	for idle in inactivity:
		events.append({"type": "inactivity", "data": idle})
	for fight in fights:
		events.append({"type": "fight", "data": fight})
	if density["level"] in {"medium", "high"}:
		events.append({"type": "density", "data": density})

	return detections, poses, events, density, falls, fights, inactivity


def main():
	args = parse_args()
	model_path, device, image_size = resolve_pose_profile(args.device, args.model, args.imgsz)
	print(f"[INFO] YOLO pose model={model_path}, device={device}, imgsz={image_size}, conf={YOLO_POSE_CONF}, iou={YOLO_POSE_IOU}")

	LOG_DIR.mkdir(parents=True, exist_ok=True)
	FRAME_DIR.mkdir(parents=True, exist_ok=True)
	CLIP_DIR.mkdir(parents=True, exist_ok=True)

	if args.source == "webcam":
		video = VideoSource.from_webcam(args.camera, target_fps=CAMERA_FPS)
	else:
		source_path = Path(args.source)
		if not source_path.exists():
			raise FileNotFoundError(f"Video file not found: {source_path}")
		video = VideoSource.from_file(str(source_path))

	pipeline = build_pipeline(args, model_path, device, image_size)
	event_logger = EventLogger(LOG_DIR, FRAME_DIR)
	action_executor = ActionExecutor(event_logger, cooldown_seconds=ACTION_COOLDOWN_SECONDS)
	clip_buffer = ClipBuffer(video.fps, CLIP_SECONDS, CLIP_DIR) if SAVE_CLIP_ON_HIGH_RISK else None

	rule_agent = RuleBasedAgent()
	if args.agent == "llm":
		try:
			agent = CrowdSafetyAgent(OPENAI_API_KEY, model=AGENT_MODEL, temperature=AGENT_TEMPERATURE)
		except Exception as exc:
			print(f"[WARN] LLM agent unavailable, using rule-based agent: {exc}")
			agent = rule_agent
	else:
		agent = rule_agent

	display = DISPLAY_WINDOW and not args.no_display
	recent_events = deque(maxlen=RECENT_EVENT_HISTORY)
	last_decision = {"risk_level": "low", "action": "none", "reasoning": "No risk"}
	risk_streak = 0
	frame_id = 0

	while True:
		frame = video.get_frame()
		if frame is None:
			break
		frame_id += 1

		detections, poses, events, density, falls, fights, inactivity = process_frame(frame, pipeline, clip_buffer)

		if events:
			for event in events:
				recent_events.append(event["type"])
				event_logger.log_event(event["type"], "info", json.dumps(event["data"]))

		risk_streak = risk_streak + 1 if (events or density["level"] != "low") else 0
		prolonged_risk = risk_streak >= RISK_HIGH_STREAK_FRAMES
		should_call_agent = events or (frame_id % AGENT_CALL_EVERY_N_FRAMES == 0)

		if should_call_agent:
			context = {
				"frame_id": frame_id,
				"timestamp": time.time(),
				"crowd": density,
				"falls": falls,
				"fights": fights,
				"inactivity": inactivity,
				"prolonged_risk": prolonged_risk,
				"recent_events": list(recent_events),
			}
			decision = analyze_with_fallback(agent, rule_agent, context)
			last_decision = decision

			if decision["risk_level"] == "high" and SAVE_SCREENSHOT_ON_HIGH_RISK:
				event_logger.log_event("risk_high", "high", decision["reasoning"], frame=frame)
				if clip_buffer and SAVE_CLIP_ON_HIGH_RISK:
					clip_path = clip_buffer.save_clip("high_risk")
					if clip_path:
						event_logger.log_event("clip_saved", "high", clip_path)

			action_executor.execute(decision, frame=frame)

		if display:
			draw_boxes(frame, detections)
			draw_poses(frame, poses)
			status_lines = [
				f"People: {density['count']}",
				f"Density: {density['density']:.2f} ({density['level']})",
				f"Risk: {last_decision['risk_level']} | Action: {last_decision['action']}",
			]
			draw_status(frame, status_lines)
			cv2.imshow("Crowd Safety", frame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

	video.release()
	if display:
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
