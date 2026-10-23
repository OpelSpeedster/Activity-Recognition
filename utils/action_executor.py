import time


class ActionExecutor:
    def __init__(self, event_logger, cooldown_seconds=10):
        self.event_logger = event_logger
        self.cooldown_seconds = cooldown_seconds
        self.last_action_time = 0

    def execute(self, decision, frame=None):
        action = decision.get("action", "none")
        risk_level = decision.get("risk_level", "low")
        reasoning = decision.get("reasoning", "")

        # This student project can notify a person, but must never pretend
        # that it has called or dispatched emergency services.
        if action not in {"none", "alert", "human_review"}:
            action = "human_review"
            reasoning = "Unsupported action was changed to human review. " + reasoning

        if action == "none":
            return

        now = time.time()
        if now - self.last_action_time < self.cooldown_seconds:
            return

        message = f"{action.upper()} | risk={risk_level} | {reasoning}"
        if action == "alert":
            print(f"[ALERT] {message}")
        elif action == "human_review":
            print(f"[HUMAN REVIEW REQUIRED] {message}")
        else:
            print(f"[ACTION] {message}")

        self.event_logger.log_event("action", risk_level, message, frame=frame)
        self.last_action_time = now
