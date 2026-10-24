"""Second-opinion verifier built from two Groq agents.

The local rule agent stays the decision authority. The verifier runs an
observation agent over the actual snapshot, hands that observation to a
separate decision agent, and returns an independent
{risk_level, action, reasoning, alert_message} verdict that is compared against
the local decision:

- agreement + high risk        -> auto-escalate to a designated reviewer
- disagreement                 -> human-in-the-loop console approval
- invalid/missing model output -> human-in-the-loop console approval

The public contract is the same as the old Modal-backed verifier (verify /
verify_encoded returning {agreement, model_decision, reason, latency_ms,
model}) so the pipeline and the decision matrix in agents/verifier.py did not
have to change. Two fields are new: `alert_message` is the WhatsApp text
composed by the decision agent, and `observation` is kept for the logs.

Never raises: any failure degrades to INSUFFICIENT so the pipeline routes to
human review instead of dying on the video loop.
"""
import time

from agents.groq_agents import (
    DEFAULT_DECISION_MODEL,
    DEFAULT_VISION_MODEL,
    GROQ_BASE_URL,
    DecisionAgent,
    VisionAgent,
    build_context_summary,
    encode_frame_jpeg,
)
from agents.verifier import INSUFFICIENT, compare_decisions

# Re-exported so existing callers and tests keep a single import site.
__all__ = [
    "GroqVerifier",
    "encode_frame_jpeg",
    "build_context_summary",
    "DEFAULT_VISION_MODEL",
    "DEFAULT_DECISION_MODEL",
]


class GroqVerifier:
    """Runs the vision agent then the decision agent over one snapshot."""

    def __init__(self, api_key, vision_model=DEFAULT_VISION_MODEL,
                 decision_model=DEFAULT_DECISION_MODEL, base_url="",
                 temperature=0.0, vision_max_tokens=700, decision_max_tokens=700,
                 timeout=10.0):
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
        self.decision = DecisionAgent(
            api_key, model=decision_model, max_tokens=decision_max_tokens, **shared
        )
        self.model = decision_model
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
        """Verifies an already-encoded JPEG.

        Takes the encoded form so callers that encode on another thread (see
        utils/verify_queue.py) never share a numpy array across threads.
        """
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
