"""Simple, predictable rules for deciding which events need human review.

This agent never contacts emergency services. The vision detectors only
produce candidates; a person must review the evidence and decide what to do.
"""


class RuleBasedAgent:
    def analyze(self, context):
        falls = context.get("falls", [])
        fights = context.get("fights", [])
        crowd = context.get("crowd", {})
        prolonged = context.get("prolonged_risk", False)

        density_level = crowd.get("level", "low")

        if falls:
            return {
                "risk_level": "high",
                "action": "human_review",
                "reasoning": "Possible fall detected; please review the clip.",
            }

        if fights and density_level in {"medium", "high"}:
            return {
                "risk_level": "high",
                "action": "human_review",
                "reasoning": "Fight-like movement in a dense area; please review the clip.",
            }

        if fights:
            return {
                "risk_level": "medium",
                "action": "human_review",
                "reasoning": "Possible fight-like movement; please review the clip.",
            }

        if density_level == "high":
            return {
                "risk_level": "medium",
                "action": "alert",
                "reasoning": "Crowd density high",
            }

        if prolonged:
            return {
                "risk_level": "medium",
                "action": "alert",
                "reasoning": "Prolonged risk state",
            }

        return {"risk_level": "low", "action": "none", "reasoning": "No risk detected"}
