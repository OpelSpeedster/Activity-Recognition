"""
Entry point: wires video capture -> local detectors -> deterministic event gate
-> event clips -> Groq review -> Telegram notification.

Run:
    python main.py --source webcam --agent rule
    python main.py --source path/to/video.mp4 --agent llm
"""
import argparse
import json
import time
from datetime import datetime
from collections import deque
from pathlib import Path

import cv2

from agents.crowd_agent import CrowdSafetyAgent
from agents.rule_agent import RuleBasedAgent
from agents.verifier import decide_outcome, needs_human_review
from agents.vlm_verifier import GroqVerifier
from config import (
	ACTION_COOLDOWN_SECONDS,
	AGENT_CALL_EVERY_N_FRAMES,
	AGENT_MODEL,
	AGENT_TEMPERATURE,
	CLIP_DIR,
	CLIP_SECONDS,
	CAMERA_FPS,
	DENSITY_AREA_SCALE,
	DENSITY_HIGH,
	DENSITY_MEDIUM,
	DISPLAY_WINDOW,
	ESCALATE_ON_AGREED_HIGH,
	FALL_ANGLE_THRESHOLD,
	FALL_MIN_FRAMES,
	FALL_MIN_DROP_RATIO,
	FIGHT_MIN_FRAMES,
	FIGHT_MOTION_THRESHOLD,
	FIGHT_NEAR_DISTANCE_RATIO,
	FIGHT_SPIKE_THRESHOLD,
	FRAME_DIR,
	GROQ_API_KEY,
	GROQ_BASE_URL,
	HORIZONTAL_RATIO_THRESHOLD,
	HITL_PROMPT_ENABLED,
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
	TELEGRAM_BOT_TOKEN,
	TELEGRAM_CHAT_ID,
	TELEGRAM_TIMEOUT_SECONDS,
	TRACK_MAX_LOST,
	VISION_MAX_TOKENS,
	VISION_MODEL,
	DECISION_MAX_TOKENS,
	DECISION_MODEL,
	VERIFY_ASYNC,
	VERIFY_ENABLED,
	VERIFY_LATENCY_TARGET_MS,
	VERIFY_MIN_HIGH_RISK_SECONDS,
	VERIFY_TEMPERATURE,
	VERIFY_TIMEOUT_SECONDS,
	YOLO_POSE_CONF,
	YOLO_POSE_IOU,
	YOLO_POSE_MAX_DET,
)
from detectors.density_estimator import DensityEstimator
from detectors.fall_detector import FallDetector
from detectors.fight_detector import FightDetector
from detectors.pose_estimator import PoseEstimator
from utils.clip_buffer import ClipBuffer, sample_event_frames
from utils.draw_utils import draw_boxes, draw_poses, draw_status
from utils.event_logger import EventLogger
from utils.simple_tracker import SimpleTracker
from utils.video_input import VideoSource
from utils.model_profile import resolve_pose_profile
from utils.telegram_notifier import TelegramNotifier
from utils.verification_logger import VerificationLogger
from utils.verify_queue import AsyncIncidentPipeline


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


def build_judge():
	"""Build the Groq event reviewer, or return None when disabled/unconfigured."""
	if not VERIFY_ENABLED:
		return None
	if not GROQ_API_KEY:
		print("[INFO] GROQ_API_KEY not set; event review will use deterministic fallback")
		return None

	try:
		return GroqVerifier(
			GROQ_API_KEY,
			vision_model=VISION_MODEL,
			decision_model=DECISION_MODEL,
			base_url=GROQ_BASE_URL,
			temperature=VERIFY_TEMPERATURE,
			vision_max_tokens=VISION_MAX_TOKENS,
			decision_max_tokens=DECISION_MAX_TOKENS,
			timeout=VERIFY_TIMEOUT_SECONDS,
		)
	except Exception as exc:
		print(f"[WARN] Verifier unavailable: {exc}")
		return None


def should_verify(judge, decision, high_risk_seconds):
	"""True when the agentic verifier should review this decision.

	Only HIGH risk that has held for VERIFY_MIN_HIGH_RISK_SECONDS reaches the
	LLMs. A brief spike (one fall frame, a momentary density jump) and all
	medium risk run the normal local path with no network call, so the agents
	only fire on a situation that actually looks persistent.

	The judge stays advisory: it never overrides the local decision, it only
	decides whether escalation needs a human confirm.
	"""
	if not judge:
		return False
	if decision.get("risk_level") != "high":
		return False
	return high_risk_seconds >= VERIFY_MIN_HIGH_RISK_SECONDS


def should_review_rule_decision(decision):
    return decision.get("action", "none") != "none"


def submit_rule_candidate(decision, incident, pipeline):
    if not should_review_rule_decision(decision):
        return False
    return pipeline.submit(incident)


def is_new_event(event_key, now, last_seen, cooldown):
    if now - last_seen.get(event_key, float("-inf")) < cooldown:
        return False
    last_seen[event_key] = now
    return True


def local_candidate_key(context):
    context = context or {}
    crowd = context.get("crowd") or {}
    candidates = [
        ("possible_fall", context.get("falls") or []),
        ("fight_like_movement", context.get("fights") or []),
        ("crowd_density", crowd.get("level") in {"medium", "high"}),
        ("inactivity", context.get("inactivity") or []),
    ]
    for alert_type, evidence in candidates:
        if evidence:
            first = evidence[0] if isinstance(evidence, list) else {}
            return alert_type, first.get("track_id", first.get("id")) if isinstance(first, dict) else None
    return "uncertain", None


def build_telegram_notifier():
    notifier = TelegramNotifier(
        TELEGRAM_BOT_TOKEN,
        TELEGRAM_CHAT_ID,
        timeout=TELEGRAM_TIMEOUT_SECONDS,
    )
    if not notifier.configured:
        print("[INFO] Telegram credentials not set; reviewer notifications are disabled")
    return notifier


def submit_completed_clips(completed, clip_buffer, incident_pipeline, event_logger):
    for clip in completed:
        try:
            samples = sample_event_frames(
                clip["frames"], clip["trigger_index"],
                clip_buffer.pre_count, clip_buffer.post_count,
            )
        except ValueError:
            samples = []
        incident = {
            **clip["metadata"],
            "event_id": clip["event_id"],
            "clip_path": clip["clip_path"],
            "frames": samples,
        }
        if not submit_rule_candidate(incident["local_decision"], incident, incident_pipeline):
            event_logger.log_event("review_queue_full", "warning", incident["event_id"])
        else:
            event_logger.log_event(
                "event_clip_ready", incident["local_decision"].get("risk_level", ""),
                incident.get("clip_path", ""),
            )


def drain_incident_results(incident_pipeline, verification_logger, event_logger):
    result = incident_pipeline.poll()
    while result is not None:
        verification_logger.log_event_result(result)
        review = result.get("review") or {}
        delivery = result.get("delivery") or {}
        if review.get("status") != "ok":
            event_logger.log_event(
                "ai_review_failed", "warning",
                f"{result.get('event_id', '')}: {review.get('reason', 'review unavailable')}",
            )
        if delivery.get("error") and delivery.get("error") != "low risk":
            event_logger.log_event(
                "telegram_delivery_failed", "warning",
                f"{result.get('event_id', '')}: {delivery['error']}",
            )
        result = incident_pipeline.poll()


def request_human_approval(local_decision, verify_result, evidence, reviewer_notifier=None):
	"""Legacy single-frame console prompt; the event-review loop does not call it."""
	print("[HUMAN REVIEW] VLM verification needs a decision.")
	print(
		f"  Local verdict : risk={local_decision['risk_level']} "
		f"action={local_decision['action']} | {local_decision['reasoning']}"
	)
	model_decision = (verify_result or {}).get("model_decision")
	if model_decision:
		print(
			f"  VLM verdict   : risk={model_decision['risk_level']} "
			f"action={model_decision['action']} | {model_decision['reasoning']}"
		)
	else:
		print(f"  VLM verdict   : unavailable | {(verify_result or {}).get('reason', '')}")
	if evidence.get("frame_path"):
		print(f"  Screenshot    : {evidence['frame_path']}")
	if evidence.get("clip_path"):
		print(f"  Clip          : {evidence['clip_path']}")

	# Legacy WhatsApp heads-up for the single-frame console approval path.
	if reviewer_notifier is not None and reviewer_notifier.request_review(
		local_decision, verify_result, evidence
	):
		print(f"  Review request sent to +{reviewer_notifier.whatsapp_number}")

	try:
		answer = input("  Type 'approve' to proceed or 'reject' to skip: ").strip().lower()
	except (EOFError, KeyboardInterrupt):
		print()
		return "rejected"
	return "approved" if answer in {"approve", "approved", "a", "y", "yes"} else "rejected"


def handle_verify_result(verify_result, decision, evidence, frame_id, notifier,
                         reviewer_notifier, action_executor, event_logger,
                         verification_logger):
	"""Legacy single-frame decision handler; the event loop uses Telegram results.

	Shared by the inline and async paths so latency warnings, HITL review,
	escalation, execution and logging behave identically no matter how the
	judge was scheduled.
	"""
	latency = verify_result.get("latency_ms", 0.0)
	if latency > VERIFY_LATENCY_TARGET_MS:
		print(f"[VERIFY] latency {latency:.0f}ms exceeds target {VERIFY_LATENCY_TARGET_MS}ms")

	human_decision = ""
	if verify_result["agreement"] == "agreement":
		print(
			f"[VERIFY] {decision['risk_level']} risk confirmed by "
			f"{verify_result.get('model') or 'verifier'} ({latency:.0f}ms)"
		)
	else:
		print(f"[VERIFY] {verify_result['agreement']}: {verify_result['reason']}")
		if needs_human_review(verify_result) and HITL_PROMPT_ENABLED:
			human_decision = request_human_approval(
				decision, verify_result, evidence, reviewer_notifier
			)
			print(f"[VERIFY] human decision: {human_decision}")

	outcome = decide_outcome(decision, verify_result, human_decision)

	# Escalate only on agreed high risk (config-gated) or explicit human approval.
	if outcome["escalate"] and (human_decision == "approved" or ESCALATE_ON_AGREED_HIGH):
		sent = notifier.escalate(
			decision,
			verify_result.get("model_decision"),
			evidence,
			alert_message=verify_result.get("alert_message", ""),
		)
		if sent:
			event_logger.log_event("escalation", decision["risk_level"], "Notified designated reviewer")

	if outcome["execute"]:
		action_executor.execute(decision)

	verification_logger.log(frame_id, decision, verify_result, human_decision, outcome["outcome"])


def handle_unverified_decision(decision, evidence, frame, notifier, action_executor, event_logger):
	"""Legacy single-frame local handler; the event loop uses Telegram results.

	Keeps local rules self-sufficient: with verification disabled, unconfigured,
	or simply not applicable (low risk), alerts and escalation behave exactly as
	they did before the judge existed.
	"""
	outcome = decide_outcome(decision, None, "")

	if outcome["escalate"] and ESCALATE_ON_AGREED_HIGH:
		if notifier.escalate(decision, None, evidence):
			event_logger.log_event("escalation", decision["risk_level"], "Notified designated reviewer")

	if outcome["execute"]:
		action_executor.execute(decision, frame=frame)


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

    detector_pipeline = build_pipeline(args, model_path, device, image_size)
    event_logger = EventLogger(LOG_DIR, FRAME_DIR)
    # Event clips are required for Telegram delivery even when the legacy
    # SAVE_CLIP_ON_HIGH_RISK option is disabled.
    clip_buffer = ClipBuffer(video.fps, CLIP_SECONDS, CLIP_DIR)
    event_verifier = build_judge()
    telegram_notifier = build_telegram_notifier()
    incident_pipeline = AsyncIncidentPipeline(event_verifier, telegram_notifier)
    verification_logger = VerificationLogger(LOG_DIR)

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
    last_seen = {}

    while True:
        frame = video.get_frame()
        if frame is None:
            break
        frame_id += 1

        detections, poses, events, density, falls, fights, inactivity = process_frame(
            frame, detector_pipeline, clip_buffer
        )
        if events:
            for event in events:
                recent_events.append(event["type"])
                event_logger.log_event(event["type"], "info", json.dumps(event["data"]))

        risk_streak = risk_streak + 1 if (events or density["level"] != "low") else 0
        prolonged_risk = risk_streak >= RISK_HIGH_STREAK_FRAMES
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
        event_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add the current frame once before triggering, so it becomes the
        # event-centered clip's final pre-roll frame.
        ready_clips = clip_buffer.add(frame)
        rule_decision = rule_agent.analyze(context)
        if should_review_rule_decision(rule_decision):
            alert_type, track_id = local_candidate_key(context)
            track_key = str(track_id) if track_id is not None else "camera"
            event_key = (alert_type, track_key)
            if is_new_event(event_key, time.monotonic(), last_seen, ACTION_COOLDOWN_SECONDS):
                event_id = f"{alert_type}-{track_key}-{frame_id}"
                metadata = {
                    "frame_id": frame_id,
                    "context": context,
                    "local_decision": rule_decision,
                    "local_alert_type": alert_type,
                    "timestamp": event_timestamp,
                }
                if clip_buffer.trigger(event_id, metadata):
                    event_logger.log_event(
                        "rule_candidate", rule_decision.get("risk_level", ""), event_id
                    )
        submit_completed_clips(ready_clips, clip_buffer, incident_pipeline, event_logger)

        should_call_agent = bool(events) or frame_id % AGENT_CALL_EVERY_N_FRAMES == 0
        if should_call_agent:
            decision = analyze_with_fallback(agent, rule_agent, context)
            last_decision = decision
            if decision.get("risk_level") == "high" and SAVE_SCREENSHOT_ON_HIGH_RISK:
                event_logger.log_event(
                    "risk_high", "high", decision.get("reasoning", ""), frame=frame
                )
                if SAVE_CLIP_ON_HIGH_RISK:
                    clip_path = clip_buffer.save_clip("high_risk")
                    if clip_path:
                        event_logger.log_event("clip_saved", "high", clip_path)

        drain_incident_results(incident_pipeline, verification_logger, event_logger)

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

    submit_completed_clips(
        clip_buffer.flush_pending(), clip_buffer, incident_pipeline, event_logger
    )
    incident_pipeline.shutdown(wait=True)
    drain_incident_results(incident_pipeline, verification_logger, event_logger)

    video.release()
    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
