"""Decision matrix for second-opinion verification.

The local rule agent stays the decision authority. A verifier (see
agents/vlm_verifier.py) returns an independent verdict that is compared
against the local decision:

- agreement + high risk        -> auto-escalate to a designated reviewer
- disagreement                 -> human-in-the-loop console approval
- invalid/missing model output -> human-in-the-loop console approval

This module holds only the pure comparison/outcome logic so it can be tested
without a network client. It never contacts emergency services; escalation
only notifies a designated human reviewer (see utils/escalation.py).
"""
import json

RISK_LEVELS = {"low", "medium", "high"}
ACTIONS = {"none", "alert", "human_review"}

AGREEMENT = "agreement"
DISAGREEMENT = "disagreement"
INSUFFICIENT = "insufficient_output"

HUMAN_APPROVED = "approved"
HUMAN_REJECTED = "rejected"


def compare_decisions(local_decision, model_decision):
    """Compares two {risk_level, action} decisions.

    Returns (status, reason) where status is one of
    AGREEMENT / DISAGREEMENT / INSUFFICIENT.
    """
    if not isinstance(model_decision, dict):
        return INSUFFICIENT, "The verification model did not return a usable verdict"

    local_risk = local_decision.get("risk_level", "low")
    local_action = local_decision.get("action", "none")
    model_risk = model_decision.get("risk_level")
    model_action = model_decision.get("action")

    differences = []
    if model_risk != local_risk:
        differences.append(f"risk_level local={local_risk} model={model_risk}")
    if model_action != local_action:
        differences.append(f"action local={local_action} model={model_action}")

    if differences:
        return DISAGREEMENT, "; ".join(differences)
    return AGREEMENT, "Local rules and model verdict match"


def needs_human_review(verify_result):
    if not verify_result:
        return False
    return verify_result.get("agreement") != AGREEMENT


def _is_high(decision):
    return isinstance(decision, dict) and decision.get("risk_level") == "high"


def decide_outcome(local_decision, verify_result, human_decision=""):
    """Applies the decision matrix.

    human_decision is '' (no prompt shown), 'approved', or 'rejected'.
    Returns {"escalate": bool, "execute": bool, "outcome": str}.
    """
    if verify_result is None:
        return {
            "escalate": _is_high(local_decision),
            "execute": True,
            "outcome": "not_verified",
        }

    status = verify_result.get("agreement")
    model_decision = verify_result.get("model_decision")

    if status == AGREEMENT:
        return {
            "escalate": _is_high(local_decision),
            "execute": True,
            "outcome": "agreement_high" if _is_high(local_decision) else "agreement",
        }

    if human_decision == HUMAN_APPROVED:
        either_high = _is_high(local_decision) or _is_high(model_decision)
        return {"escalate": either_high, "execute": True, "outcome": "human_approved"}

    if human_decision == HUMAN_REJECTED:
        return {"escalate": False, "execute": False, "outcome": "human_rejected"}

    # No approval channel: never escalate on an unreviewed mismatch.
    return {"escalate": False, "execute": True, "outcome": status or "unknown"}


def parse_verdict(text):
    """Extracts a verdict dict from raw model text. Raises ValueError on junk."""
    return _normalize_output(_safe_parse(text))


def _safe_parse(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass

    text = str(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except ValueError:
            pass
    raise ValueError("The verification model did not return valid JSON")


def _normalize_output(data):
    if not isinstance(data, dict):
        raise ValueError("The verification model response must be a JSON object")

    risk_level = str(data.get("risk_level", "")).strip().lower()
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"Invalid risk_level from verification model: {risk_level!r}")

    action = str(data.get("action", "")).strip().lower()
    if action not in ACTIONS:
        raise ValueError(f"Invalid action from verification model: {action!r}")

    reasoning = str(data.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("The verification model returned an empty reasoning")

    return {"risk_level": risk_level, "action": action, "reasoning": reasoning}
