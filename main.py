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
from agents.verifier import decide_outcome, needs_human_review
from agents.vlm_verifier import GroqVerifier
from config import (
	ACTION_COOLDOWN_SECONDS,
	AGENT_CALL_EVERY_N_FRAMES,
	AGENT_MODEL,
	AGENT_TEMPERATURE,
	AUTHORITY_PHONE_NUMBER,
	AUTHORITY_WHATSAPP_NUMBER,
	CLIP_DIR,
	CLIP_SECONDS,
	CAMERA_FPS,
	DENSITY_AREA_SCALE,
	DENSITY_HIGH,
	DENSITY_MEDIUM,
	DISPLAY_WINDOW,
	ESCALATE_ON_AGREED_HIGH,
	ESCALATION_COOLDOWN_SECONDS,
	ESCALATION_DRY_RUN,
	EVIDENCE_PUBLIC_HOST,
	EVIDENCE_SERVER_ENABLED,
	EVIDENCE_SERVER_PORT,
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
	REVIEW_REQUEST_COOLDOWN_SECONDS,
	REVIEWER_WHATSAPP_NUMBER,
	RISK_HIGH_STREAK_FRAMES,
	SAVE_CLIP_ON_HIGH_RISK,
	SAVE_SCREENSHOT_ON_HIGH_RISK,
	TRACK_MAX_DISTANCE_RATIO,
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
from utils.action_executor import ActionExecutor
from utils.clip_buffer import ClipBuffer
from utils.draw_utils import draw_boxes, draw_poses, draw_status
from utils.escalation import EscalationNotifier, ReviewerNotifier
from utils.evidence_server import EvidenceServer
from utils.event_logger import EventLogger
from utils.simple_tracker import SimpleTracker
from utils.video_input import VideoSource
from utils.model_profile import resolve_pose_profile
from utils.verification_logger import VerificationLogger
from utils.verify_queue import AsyncVerifier


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
	"""Builds the Groq two-agent verifier, or returns None to skip it.

	A vision agent (Qwen) looks at the snapshot, a decision agent (GPT-OSS)
	turns that into an independent verdict plus the WhatsApp alert text. The
	local rules stay the authority; this is only the second opinion.
	"""
	if not VERIFY_ENABLED:
		return None
	if not GROQ_API_KEY:
		print("[INFO] GROQ_API_KEY not set; second-opinion verification disabled")
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


def request_human_approval(local_decision, verify_result, evidence, reviewer_notifier=None):
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

	# Send-only heads-up to the reviewer's WhatsApp; they answer in the console.
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
	"""Applies the decision matrix to one finished verdict.

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
	"""Runs the pre-verification behavior for a decision no judge reviewed.

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

	judge = build_judge()
	async_verifier = AsyncVerifier(judge) if (judge and VERIFY_ASYNC) else None
	if async_verifier is not None:
		print("[INFO] Verifier running async; the video loop will not stall")

	# Optional LAN server so the wa.me message carries a photo link, not a path.
	evidence_server = EvidenceServer(
		FRAME_DIR, port=EVIDENCE_SERVER_PORT, public_host=EVIDENCE_PUBLIC_HOST
	)
	evidence_base_url = ""
	if EVIDENCE_SERVER_ENABLED and evidence_server.start():
		evidence_base_url = (
			f"http://{EVIDENCE_PUBLIC_HOST}:{evidence_server.bound_port}"
			if EVIDENCE_PUBLIC_HOST
			else ""
		)
		if EVIDENCE_SERVER_ENABLED and not EVIDENCE_PUBLIC_HOST:
			print(
				"[WARN] EVIDENCE_PUBLIC_HOST not set; escalation messages will "
				"carry file paths instead of photo links"
			)

	notifier = EscalationNotifier(
		AUTHORITY_WHATSAPP_NUMBER,
		AUTHORITY_PHONE_NUMBER,
		cooldown_seconds=ESCALATION_COOLDOWN_SECONDS,
		dry_run=ESCALATION_DRY_RUN,
		evidence_base_url=evidence_base_url,
	)
	if not notifier.configured:
		print("[INFO] Authority numbers not set; escalation notifications are disabled")

	reviewer_notifier = ReviewerNotifier(
		REVIEWER_WHATSAPP_NUMBER,
		cooldown_seconds=REVIEW_REQUEST_COOLDOWN_SECONDS,
		dry_run=ESCALATION_DRY_RUN,
		evidence_base_url=evidence_base_url,
	)
	if not reviewer_notifier.configured:
		print("[INFO] Reviewer number not set; review requests go to the console only")

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
	high_risk_streak = 0
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
		# How long the most recent decision has been high risk. This is what
		# gates the agentic verifier, so a one-frame spike costs nothing and
		# only a sustained high-risk situation reaches the LLMs.
		high_risk_streak = high_risk_streak + 1 if last_decision["risk_level"] == "high" else 0
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

			# Save evidence BEFORE verification so escalation can reference it.
			evidence = {"frame_path": "", "clip_path": ""}
			if decision["risk_level"] == "high" and SAVE_SCREENSHOT_ON_HIGH_RISK:
				evidence["frame_path"] = event_logger.log_event("risk_high", "high", decision["reasoning"], frame=frame)
				if clip_buffer and SAVE_CLIP_ON_HIGH_RISK:
					clip_path = clip_buffer.save_clip("high_risk")
					if clip_path:
						evidence["clip_path"] = clip_path
						event_logger.log_event("clip_saved", "high", clip_path)

			# Second-opinion verification is gated on sustained high risk; see
			# should_verify(). Everything else -- medium risk, a brief spike,
			# or no risk at all -- runs the normal local path with no network
			# call.
			high_risk_seconds = high_risk_streak / max(CAMERA_FPS, 1)
			verified_here = should_verify(judge, decision, high_risk_seconds)
			if verified_here:
				if async_verifier is not None:
					# Non-blocking: verdict is picked up on a later frame.
					async_verifier.submit(frame_id, frame, context, decision, evidence)
				else:
					handle_verify_result(
						judge.verify(frame, context, decision),
						decision,
						evidence,
						frame_id,
						notifier,
						reviewer_notifier,
						action_executor,
						event_logger,
						verification_logger,
					)

			if not verified_here:
				handle_unverified_decision(
					decision, evidence, frame, notifier, action_executor, event_logger
				)

		# Drain any verdict the background judge has already finished. The job
		# carries its own frame_id/decision/evidence, so it is judged against
		# the frame that produced it, not the current one.
		if async_verifier is not None:
			finished = async_verifier.poll()
			if finished is not None:
				handle_verify_result(
					finished["result"],
					finished["decision"],
					finished["evidence"],
					finished["frame_id"],
					notifier,
					reviewer_notifier,
					action_executor,
					event_logger,
					verification_logger,
				)

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

	# Collect verdicts still in flight. File runs would otherwise drop the
	# last few events, since their decisions never got a frame to land on.
	if async_verifier is not None:
		# Two sequential agent calls (vision, then decision) share one job.
		drain_deadline = time.time() + (2 * VERIFY_TIMEOUT_SECONDS) + 1.0
		while time.time() < drain_deadline:
			finished = async_verifier.poll()
			if finished is not None:
				handle_verify_result(
					finished["result"],
					finished["decision"],
					finished["evidence"],
					finished["frame_id"],
					notifier,
					reviewer_notifier,
					action_executor,
					event_logger,
					verification_logger,
				)
				continue
			if not async_verifier.in_flight:
				break
			time.sleep(0.05)
		async_verifier.shutdown(wait=False)

	video.release()
	if display:
		cv2.destroyAllWindows()
	evidence_server.stop()


if __name__ == "__main__":
	main()
