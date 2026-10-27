"""
Central configuration for the crowd-safety HAR system.

All values are overridable via a `.env` file (see `.env.example`).
Grouped by subsystem so it's easy to find what tunes what.
"""
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
FRAME_DIR = BASE_DIR / "frames"
CLIP_DIR = BASE_DIR / "clips"


def _get_bool(name, default):
    return os.getenv(name, "1" if default else "0") == "1"


def _get_int(name, default):
    return int(os.getenv(name, str(default)))


def _get_float(name, default):
    return float(os.getenv(name, str(default)))


# ---------------------------------------------------------------------------
# Camera / video input
# ---------------------------------------------------------------------------
CAMERA_FPS = _get_int("CAMERA_FPS", 30)  # 60 was unrealistic for most webcams

# ---------------------------------------------------------------------------
# YOLO-Pose model. CUDA desktops use a larger profile; CPU-only systems use
# the nano profile. YOLO_POSE_MODEL can override the selected model explicitly.
# ---------------------------------------------------------------------------
YOLO_POSE_MODEL = os.getenv("YOLO_POSE_MODEL", "").strip()
YOLO_POSE_MODEL_CPU = os.getenv("YOLO_POSE_MODEL_CPU", "yolov8n-pose.pt")
YOLO_POSE_MODEL_CUDA = os.getenv("YOLO_POSE_MODEL_CUDA", "yolov8m-pose.pt")
YOLO_POSE_CONF = _get_float("YOLO_POSE_CONF", 0.10)
YOLO_POSE_IOU = _get_float("YOLO_POSE_IOU", 0.65)
YOLO_POSE_IMGSZ_CPU = _get_int("YOLO_POSE_IMGSZ_CPU", 640)
YOLO_POSE_IMGSZ_CUDA = _get_int("YOLO_POSE_IMGSZ_CUDA", 960)
YOLO_POSE_MAX_DET = _get_int("YOLO_POSE_MAX_DET", 1000)
KEYPOINT_CONF_THRESHOLD = _get_float("KEYPOINT_CONF_THRESHOLD", 0.5)

# ---------------------------------------------------------------------------
# Fall detection
# ---------------------------------------------------------------------------
FALL_ANGLE_THRESHOLD = _get_float("FALL_ANGLE_THRESHOLD", 55)
FALL_MIN_FRAMES = _get_int("FALL_MIN_FRAMES", 5)
FALL_MIN_DROP_RATIO = _get_float("FALL_MIN_DROP_RATIO", 0.35)  # sudden height collapse
INACTIVITY_FRAMES = _get_int("INACTIVITY_FRAMES", 15)
INACTIVITY_MOTION_THRESHOLD = _get_float("INACTIVITY_MOTION_THRESHOLD", 0.01)
HORIZONTAL_RATIO_THRESHOLD = _get_float("HORIZONTAL_RATIO_THRESHOLD", 1.3)

# ---------------------------------------------------------------------------
# Fight detection
# ---------------------------------------------------------------------------
FIGHT_MOTION_THRESHOLD = _get_float("FIGHT_MOTION_THRESHOLD", 0.035)
FIGHT_SPIKE_THRESHOLD = _get_float("FIGHT_SPIKE_THRESHOLD", 0.06)
FIGHT_NEAR_DISTANCE_RATIO = _get_float("FIGHT_NEAR_DISTANCE_RATIO", 0.12)
FIGHT_MIN_FRAMES = _get_int("FIGHT_MIN_FRAMES", 3)  # require sustained motion, not a 1-frame spike
MOTION_HISTORY = _get_int("MOTION_HISTORY", 5)

# ---------------------------------------------------------------------------
# Density estimation
# ---------------------------------------------------------------------------
DENSITY_AREA_SCALE = _get_float("DENSITY_AREA_SCALE", 100000)
DENSITY_MEDIUM = _get_float("DENSITY_MEDIUM", 8)
DENSITY_HIGH = _get_float("DENSITY_HIGH", 15)

# ---------------------------------------------------------------------------
# Decision agent (rule-based fallback / LLM)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AGENT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AGENT_TEMPERATURE = _get_float("AGENT_TEMPERATURE", 0.2)
AGENT_CALL_EVERY_N_FRAMES = _get_int("AGENT_CALL_EVERY_N_FRAMES", 10)

RISK_HIGH_STREAK_FRAMES = _get_int("RISK_HIGH_STREAK_FRAMES", 30)
ACTION_COOLDOWN_SECONDS = _get_int("ACTION_COOLDOWN_SECONDS", 10)

# ---------------------------------------------------------------------------
# Groq event reviewer. The deterministic RuleBasedAgent gates event review;
# Qwen sees three sampled event frames plus detector context, then GPT-OSS
# returns alert type, risk, and summary. Both models share GROQ_API_KEY.
# The MP4 stays local until Telegram sends it to the configured human reviewer.
# VERIFY_ENABLED disables Groq review but keeps deterministic fallback alerts.
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip().rstrip("/")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b").strip()
DECISION_MODEL = os.getenv("DECISION_MODEL", "openai/gpt-oss-120b").strip()
VERIFY_TEMPERATURE = _get_float("VERIFY_TEMPERATURE", 0.0)
VISION_MAX_TOKENS = _get_int("VISION_MAX_TOKENS", 700)
DECISION_MAX_TOKENS = _get_int("DECISION_MAX_TOKENS", 700)
# Per-request timeout; a Groq failure routes the deterministic candidate to Telegram.
VERIFY_TIMEOUT_SECONDS = _get_float("VERIFY_TIMEOUT_SECONDS", 10.0)
# Warn when a completed verification exceeds this. Groq runs at ~450-500 tps,
# so both agents together should land well under a second of decode time.
VERIFY_LATENCY_TARGET_MS = _get_int("VERIFY_LATENCY_TARGET_MS", 2000)
VERIFY_ENABLED = _get_bool("VERIFY_ENABLED", True)
# Legacy single-frame verifier threshold; the event-review path is gated by
# a non-none deterministic rule action and does not use this setting.
VERIFY_MIN_HIGH_RISK_SECONDS = _get_float("VERIFY_MIN_HIGH_RISK_SECONDS", 2.0)
# Legacy single-frame verifier setting; event review always uses a worker queue.
VERIFY_ASYNC = _get_bool("VERIFY_ASYNC", True)

# ---------------------------------------------------------------------------
# Legacy manual-link escalation settings. The live event-review route uses
# Telegram and does not open WhatsApp/tel links or ask for console approval.
# ---------------------------------------------------------------------------
AUTHORITY_WHATSAPP_NUMBER = os.getenv("AUTHORITY_WHATSAPP_NUMBER", "").strip()
AUTHORITY_PHONE_NUMBER = os.getenv("AUTHORITY_PHONE_NUMBER", "").strip()
REVIEWER_WHATSAPP_NUMBER = os.getenv("REVIEWER_WHATSAPP_NUMBER", "").strip()
ESCALATE_ON_AGREED_HIGH = _get_bool("ESCALATE_ON_AGREED_HIGH", True)
ESCALATION_COOLDOWN_SECONDS = _get_int("ESCALATION_COOLDOWN_SECONDS", 30)
ESCALATION_DRY_RUN = _get_bool("ESCALATION_DRY_RUN", False)
HITL_PROMPT_ENABLED = _get_bool("HITL_PROMPT_ENABLED", True)
# Reuse a cooldown so a repeating failure cannot spam the reviewer.
REVIEW_REQUEST_COOLDOWN_SECONDS = _get_int("REVIEW_REQUEST_COOLDOWN_SECONDS", 60)

# ---------------------------------------------------------------------------
# Telegram reviewer notifications
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_TIMEOUT_SECONDS = _get_int("TELEGRAM_TIMEOUT_SECONDS", 10)

# ---------------------------------------------------------------------------
# Legacy evidence-server settings for manual-link helpers; unused by the
# Telegram event-review route.
# ---------------------------------------------------------------------------
EVIDENCE_SERVER_ENABLED = _get_bool("EVIDENCE_SERVER_ENABLED", False)
EVIDENCE_SERVER_PORT = _get_int("EVIDENCE_SERVER_PORT", 8090)
EVIDENCE_PUBLIC_HOST = os.getenv("EVIDENCE_PUBLIC_HOST", "").strip()

# ---------------------------------------------------------------------------
# Output / display
# ---------------------------------------------------------------------------
DISPLAY_WINDOW = _get_bool("DISPLAY_WINDOW", True)
SAVE_SCREENSHOT_ON_HIGH_RISK = _get_bool("SAVE_SCREENSHOT_ON_HIGH_RISK", True)
SAVE_CLIP_ON_HIGH_RISK = _get_bool("SAVE_CLIP_ON_HIGH_RISK", True)
CLIP_SECONDS = _get_int("CLIP_SECONDS", 3)

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
TRACK_MAX_DISTANCE_RATIO = _get_float("TRACK_MAX_DISTANCE_RATIO", 0.1)
TRACK_MAX_LOST = _get_int("TRACK_MAX_LOST", 30)

RECENT_EVENT_HISTORY = _get_int("RECENT_EVENT_HISTORY", 10)
