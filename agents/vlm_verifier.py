"""Single-model Groq event reviewer with legacy single-frame compatibility.

The event path sends three sampled frames and detector context to Qwen once.
Its structured decision is returned to the asynchronous Telegram pipeline.
"""
import time

from agents.groq_agents import (
    DEFAULT_VISION_MODEL,
    GROQ_BASE_URL,
    DecisionAgent,
    VisionAgent,
    build_context_summary,
    encode_frame_jpeg,
)
from agents.verifier import INSUFFICIENT, compare_decisions

__all__ = [
    "GroqVerifier",
    "encode_frame_jpeg",
    "build_context_summary",
    "DEFAULT_VISION_MODEL",
]


class GroqVerifier:
    """Runs Qwen event review and the legacy single-frame compatibility path."""

    def __init__(self, api_key, vision_model=DEFAULT_VISION_MODEL, base_url="",
                 temperature=0.0, vision_max_tokens=256, timeout=10.0):
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")
        shared = {
            "base_url": base_url or GROQ_BASE_URL,
            "temperature": temperature,
            "timeout": timeout,
        }
        self.vision = VisionAgent(
            api_key, model=vision_model, max_tokens=vision_max_tokens, **shared
        )
        # The compatibility path also uses Qwen; the event path below only
        # calls the vision-language judge once.
        self.decision = DecisionAgent(
            api_key, model=vision_model, max_tokens=vision_max_tokens, **shared
        )
        self.model = vision_model
        self.vision_model = vision_model

    def _result(self, status, model_decision, reason, started, observation=None):
        return {
            "agreement": status,
            "model_decision": model_decision,
            "reason": reason,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "model": self.model,
            "vision_model": self.vision_model,
            "alert_message": (model_decision or {}).get("alert_message", ""),
            "observation": observation,
        }

    def verify(self, frame, context, local_decision):
        """Encodes the frame, then verifies. See verify_encoded for the shape."""
        return self.verify_encoded(encode_frame_jpeg(frame), context, local_decision)

    def verify_encoded(self, frame_jpeg_b64, context, local_decision):
        """Compatibility verifier for older single-frame callers."""
        started = time.perf_counter()
        if not frame_jpeg_b64:
            return self._result(
                INSUFFICIENT,
                None,
                "Snapshot could not be encoded for the verifier",
                started,
            )

        try:
            observation = self.vision.observe_encoded(frame_jpeg_b64, context)
        except Exception as exc:
            return self._result(
                INSUFFICIENT,
                None,
                f"Vision agent call failed: {exc}",
                started,
            )

        try:
            model_decision = self.decision.decide(observation, context)
            status, reason = compare_decisions(local_decision, model_decision)
        except Exception as exc:
            return self._result(
                INSUFFICIENT,
                None,
                f"Decision agent call failed: {exc}",
                started,
                observation,
            )

        return self._result(status, model_decision, reason, started, observation)

    def review_event_encoded(self, frame_jpegs_b64, context):
        """Have Qwen judge three event frames in one Groq API request."""
        started = time.perf_counter()
        try:
            model_decision = self.vision.classify_event_encoded(
                frame_jpegs_b64, context
            )
        except Exception as exc:
            return {
                "status": "error",
                "model_decision": None,
                "reason": str(exc),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "model": self.model,
            }
        return {
            "status": "ok",
            "model_decision": model_decision,
            "reason": "",
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "model": self.model,
        }
