"""Optional LLM helper that summarizes detector results for a reviewer.

It receives event counts and measurements, not video frames. It cannot
contact emergency services. If its response is invalid, main.py falls back
to the deterministic RuleBasedAgent.
"""
import json

from openai import OpenAI
from agents.rule_agent import RuleBasedAgent


class CrowdSafetyAgent:
    def __init__(self, api_key, model="gpt-4o-mini", temperature=0.2):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _build_prompt(self, context):
        crowd = context.get("crowd", {})
        falls = context.get("falls", [])
        fights = context.get("fights", [])
        inactivity = context.get("inactivity", [])
        prolonged = context.get("prolonged_risk", False)
        recent = context.get("recent_events", [])

        lines = []
        if falls:
            lines.append(f"Possible fall candidates: {len(falls)}")
        if fights:
            lines.append(f"Fight-like movement candidates: {len(fights)}")
        if inactivity:
            lines.append(f"Inactivity alerts: {len(inactivity)}")
        if not lines:
            lines.append("No candidate events were reported by the detectors.")

        lines.append(
            f"Crowd density: {crowd.get('density', 0):.2f} (level: {crowd.get('level', 'low')})"
        )
        lines.append(f"People detected: {crowd.get('count', 0)}")
        lines.append(f"Prolonged risk: {prolonged}")

        recent_text = ", ".join(recent[-5:]) if recent else "none"

        prompt = (
            "Summarize these computer-vision candidate events for a human reviewer.\n"
            "Use only the supplied measurements. Do not infer a person's intent, identity, or guilt.\n"
            "Treat fall and fight outputs as uncertain candidates. Never recommend calling or dispatching emergency services.\n"
            "Return ONLY JSON in this format:\n"
            "{\"reasoning\": \"short factual review note\"}\n\n"
            "Situation:\n"
            f"- {'; '.join(lines)}\n"
            f"- Recent events: {recent_text}\n"
        )

        return prompt

    @staticmethod
    def _safe_parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        raise ValueError("The AI agent did not return valid JSON")

    @staticmethod
    def _normalize_output(data):
        if not isinstance(data, dict):
            raise ValueError("The AI agent response must be a JSON object")

        reasoning = str(data.get("reasoning", "")).strip()
        if not reasoning:
            raise ValueError("The AI agent returned an empty review note")
        return {"reasoning": reasoning}

    def analyze(self, context):
        prompt = self._build_prompt(context)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize candidate events for human review. Do not choose actions or infer intent.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        content = response.choices[0].message.content or "{}"
        data = self._safe_parse(content)
        llm_suggestion = self._normalize_output(data)

        # The LLM may make mistakes. Keep the actual risk/action decision in
        # simple local rules; use the LLM only to add a short review note.
        decision = RuleBasedAgent().analyze(context)
        decision["reasoning"] = llm_suggestion["reasoning"]
        return decision
