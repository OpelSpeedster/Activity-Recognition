"""Groq agents for three-frame event observation and structured decisions.

Both models share one GROQ_API_KEY through the OpenAI-compatible Groq endpoint.
Qwen describes what is visible across before/event/after frames; GPT-OSS turns
that observation and the deterministic detector context into an alert type,
risk level, and short summary. Legacy single-frame verifier methods remain
available for compatibility with older callers.
"""
import base64

from openai import OpenAI

from agents.verifier import ACTIONS, RISK_LEVELS

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"
DEFAULT_DECISION_MODEL = "openai/gpt-oss-120b"

CONFIDENCE_LEVELS = {"high", "medium", "low"}
EVENT_TYPES = {"possible_fall", "fight_like_movement", "crowd_density", "inactivity", "uncertain"}
EVENT_RISK_LEVELS = {"low", "medium", "high"}

# qwen3.8-27b only supports strict:false, so the schema is advisory and the
# tolerant parser below still has to cope with the odd malformed reply.
VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "activity": {"type": "string"},
        "people_count": {"type": "integer"},
        "risk_signals": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "notes": {"type": "string"},
    },
    "required": [
        "scene", "activity", "people_count", "risk_signals", "confidence", "notes",
    ],
}

# openai/gpt-oss-120b supports strict:true, which uses constrained decoding to
# guarantee the reply matches, so a malformed verdict is not a failure mode.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_level": {"type": "string", "enum": sorted(RISK_LEVELS)},
        "action": {"type": "string", "enum": sorted(ACTIONS)},
        "reasoning": {"type": "string"},
        "alert_message": {"type": "string"},
    },
    "required": ["risk_level", "action", "reasoning", "alert_message"],
}

VISION_SYSTEM_PROMPT = (
    "You are the observation agent of a crowd-safety system. You receive one "
    "or more CCTV frames in chronological order, plus raw measurements from "
    "on-site detectors. When three frames are provided, they show before, at, "
    "and after a candidate event. Describe only what is visibly present and "
    "how it changes across the frames. Treat detector measurements as uncertain "
    "candidates to check, not as facts. Report hazards only when visible. Do "
    "not identify anyone; do not infer a person's intent, identity, or guilt. Do not choose a "
    "risk level or an action; a separate agent does that. Return only JSON with "
    "scene, activity, people_count, risk_signals, confidence, and notes."
)

EVENT_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "alert_type": {"type": "string", "enum": sorted(EVENT_TYPES)},
        "risk_level": {"type": "string", "enum": sorted(EVENT_RISK_LEVELS)},
        "summary": {"type": "string"},
    },
    "required": ["alert_type", "risk_level", "summary"],
}

EVENT_DECISION_SYSTEM_PROMPT = (
    "You are the decision agent for a human-reviewed safety alert system. "
    "Use the visual observation and detector context to choose exactly one "
    "alert type: possible_fall, fight_like_movement, crowd_density, "
    "inactivity, or uncertain. Choose risk low, medium, or high. Ground the "
    "decision in visible evidence, keep the summary concise and factual, and "
    "do not identify people, assign blame, or recommend emergency services. "
    "Return only JSON with alert_type, risk_level, and summary."
)

DECISION_SYSTEM_PROMPT = (
    "You are the decision agent of a crowd-safety system. You are given an "
    "observation produced by a separate vision agent that looked at the "
    "snapshot, plus the raw measurements from the on-site detectors.\n"
    "You are deliberately not shown the on-site system's own verdict, so your "
    "judgement must be independent and evidence-led. Weigh the observation "
    "more heavily than the detector measurements, since only the observation "
    "comes from actually looking at the frame.\n"
    "Never recommend calling or dispatching emergency services. This system "
    "only notifies a designated human reviewer.\n"
    "Write alert_message as the plain-language incident summary a human reads "
    "first in a human-review notification: what was seen, why it is a concern, and what "
    "should happen next. Keep it under 60 words, factual, and free of blame or "
    "speculation about who anyone is. When the risk is low, say so in one "
    "sentence.\n"
    "Return only a JSON object with risk_level (low|medium|high), action "
    "(none|alert|human_review), reasoning (short factual justification), "
    "alert_message."
)


def encode_frame_jpeg(frame, quality=85):
    """Encodes a BGR ndarray to a base64 JPEG string. Returns '' on failure."""
    if frame is None:
        return ""
    try:
        import cv2

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        return base64.b64encode(buffer.tobytes()).decode("ascii")
    except Exception:
        return ""


def build_context_summary(context):
    """Renders the detector context as the text half of the prompt.

    Deliberately excludes the local decision so neither agent can simply echo
    it back -- the whole point is an independent verdict.
    """
    crowd = context.get("crowd", {})
    falls = context.get("falls", [])
    fights = context.get("fights", [])
    inactivity = context.get("inactivity", [])
    prolonged = context.get("prolonged_risk", False)
    recent = context.get("recent_events", [])

    lines = []
    if falls:
        lines.append(
            f"Possible fall candidates: {len(falls)} (details: {str(falls)[:400]})"
        )
    if fights:
        lines.append(
            f"Fight-like movement candidates: {len(fights)} (details: {str(fights)[:400]})"
        )
    if inactivity:
        lines.append(
            f"Inactivity alerts: {len(inactivity)} (details: {str(inactivity)[:400]})"
        )
    if not lines:
        lines.append("No candidate events were reported by the detectors.")

    lines.append(
        f"Crowd density: {crowd.get('density', 0):.2f} "
        f"(level: {crowd.get('level', 'low')})"
    )
    lines.append(f"People detected: {crowd.get('count', 0)}")
    lines.append(f"Prolonged risk: {prolonged}")
    recent_text = ", ".join(recent[-5:]) if recent else "none"
    lines.append(f"Recent events: {recent_text}")

    return "Situation reported by the detectors:\n" + "\n".join(
        f"- {line}" for line in lines
    )


def image_message(frame_jpeg_b64, text):
    """Build a user message with one or more JPEG data-URI image blocks."""
    images = [frame_jpeg_b64] if isinstance(frame_jpeg_b64, str) else list(frame_jpeg_b64 or [])
    if len(images) == 3:
        text = text + "\nImages are in order: before the event, event frame, after the event."
    content = [{"type": "text", "text": text}]
    for encoded in images:
        if encoded:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
    return content


def parse_json_reply(content):
    """Extracts a JSON object from raw model text. Raises ValueError on junk."""
    import json

    text = str(content or "")
    try:
        return json.loads(text)
    except ValueError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except ValueError:
            pass
    raise ValueError("The model did not return valid JSON")


def parse_observation(content):
    """Validates the vision agent's reply into an observation dict."""
    data = parse_json_reply(content)
    if not isinstance(data, dict):
        raise ValueError("The vision agent response must be a JSON object")

    confidence = str(data.get("confidence", "")).strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"Invalid confidence from vision agent: {confidence!r}")

    signals = data.get("risk_signals", [])
    if isinstance(signals, str):
        signals = [signals]
    if not isinstance(signals, list):
        raise ValueError("risk_signals from the vision agent must be a list")

    return {
        "scene": str(data.get("scene", "")).strip(),
        "activity": str(data.get("activity", "")).strip(),
        "people_count": int(data.get("people_count", 0) or 0),
        "risk_signals": [str(s).strip() for s in signals if str(s).strip()],
        "confidence": confidence,
        "notes": str(data.get("notes", "")).strip(),
    }


def parse_decision(content):
    """Validates the decision agent's reply into a verdict + alert message."""
    data = parse_json_reply(content)
    if not isinstance(data, dict):
        raise ValueError("The decision agent response must be a JSON object")

    risk_level = str(data.get("risk_level", "")).strip().lower()
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"Invalid risk_level from decision agent: {risk_level!r}")

    action = str(data.get("action", "")).strip().lower()
    if action not in ACTIONS:
        raise ValueError(f"Invalid action from decision agent: {action!r}")

    reasoning = str(data.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("The decision agent returned an empty reasoning")

    return {
        "risk_level": risk_level,
        "action": action,
        "reasoning": reasoning,
        "alert_message": str(data.get("alert_message", "")).strip(),
    }


def parse_event_decision(content):
    """Validate the structured event classification returned by GPT-OSS."""
    data = parse_json_reply(content)
    if not isinstance(data, dict):
        raise ValueError("The event decision must be a JSON object")

    alert_type = str(data.get("alert_type", "")).strip().lower()
    if alert_type not in EVENT_TYPES:
        raise ValueError(f"Invalid alert_type from decision agent: {alert_type!r}")

    risk_level = str(data.get("risk_level", "")).strip().lower()
    if risk_level not in EVENT_RISK_LEVELS:
        raise ValueError(f"Invalid risk_level from decision agent: {risk_level!r}")

    summary = str(data.get("summary", "")).strip()
    if not summary:
        raise ValueError("The decision agent returned an empty summary")

    return {"alert_type": alert_type, "risk_level": risk_level, "summary": summary}


def render_observation(observation):
    """Flattens an observation dict into the text the decision agent reads."""
    signals = observation.get("risk_signals") or []
    hazard_text = ", ".join(signals) if signals else "none observed"
    return (
        f"Observed scene: {observation.get('scene', 'unknown')}\n"
        f"Observed activity: {observation.get('activity', 'unknown')}\n"
        f"Estimated people visible: {observation.get('people_count', 0)}\n"
        f"Hazards the vision agent could see: {hazard_text}\n"
        f"Vision agent confidence: {observation.get('confidence', 'low')}\n"
        f"Vision agent caveats: {observation.get('notes', 'none')}"
    )


def _response_format(name, schema, strict):
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": strict, "schema": schema},
    }


class _GroqAgent:
    """Shared OpenAI-SDK client bound to Groq."""

    def __init__(self, api_key, model, base_url="", temperature=0.0,
                 max_tokens=700, timeout=8.0):
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = (base_url or GROQ_BASE_URL).rstrip("/")
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout)

    def create(self, messages, schema, strict):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=_response_format(self.model.replace("/", "_"), schema, strict),
        )
        return response.choices[0].message.content or "{}"


class VisionAgent(_GroqAgent):
    """Looks at event frames and returns factual, decision-free observations."""

    def __init__(self, api_key, model=DEFAULT_VISION_MODEL, **kwargs):
        super().__init__(api_key, model, **kwargs)

    def build_messages(self, frame_jpeg_b64, context):
        text = build_context_summary(context)
        return [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": image_message(frame_jpeg_b64, text)},
        ]

    def observe_encoded(self, frame_jpeg_b64, context):
        """Compatibility method for the existing one-image verifier."""
        content = self.create(
            self.build_messages(frame_jpeg_b64, context), VISION_SCHEMA, strict=False
        )
        return parse_observation(content)

    def observe_frames_encoded(self, frame_jpegs_b64, context):
        """Observe exactly three event-centered images in chronological order."""
        images = list(frame_jpegs_b64 or [])
        if len(images) != 3 or any(not image for image in images):
            raise ValueError("event review requires three encoded frames")
        content = self.create(
            self.build_messages(images, context), VISION_SCHEMA, strict=False
        )
        return parse_observation(content)

class DecisionAgent(_GroqAgent):
    """Produces a legacy verdict or a typed event classification."""

    def __init__(self, api_key, model=DEFAULT_DECISION_MODEL, **kwargs):
        super().__init__(api_key, model, **kwargs)

    def build_messages(self, observation, context):
        text = build_context_summary(context)
        return [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": render_observation(observation) + "\n\n" + text,
            },
        ]

    def decide(self, observation, context):
        """Returns the legacy {risk_level, action, reasoning, alert_message}."""
        content = self.create(
            self.build_messages(observation, context), DECISION_SCHEMA, strict=True
        )
        return parse_decision(content)

    def build_event_messages(self, observation, context):
        text = build_context_summary(context)
        return [
            {"role": "system", "content": EVENT_DECISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": render_observation(observation) + "\n\n" + text,
            },
        ]

    def classify_event(self, observation, context):
        """Return {alert_type, risk_level, summary} with schema validation."""
        content = self.create(
            self.build_event_messages(observation, context),
            EVENT_DECISION_SCHEMA,
            strict=True,
        )
        return parse_event_decision(content)
