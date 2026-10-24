"""Notify designated humans via wa.me and tel: links.

This is NOT emergency dispatch. Both channels open a WhatsApp chat with a
pre-filled message (and the dialer for the phone number), so each action
requires one manual click by the operator.

Two distinct recipients:

- EscalationNotifier -> AUTHORITY_*: the high-risk escalation, carrying the
  verdicts and, when the evidence server is up, a tappable link to the photo.
- ReviewerNotifier   -> REVIEWER_*: the "please evaluate this" request raised
  when the VLM disagrees or fails. The reviewer answers via the console
  prompt; wa.me is send-only.
"""
import time
import urllib.parse
import webbrowser


def _digits(value):
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def build_wa_link(number, message):
    return f"https://wa.me/{number}?text={urllib.parse.quote(message)}"


class EscalationNotifier:
    def __init__(self, whatsapp_number="", phone_number="", cooldown_seconds=30,
                 opener=None, dry_run=False, evidence_base_url=""):
        self.whatsapp_number = _digits(whatsapp_number)
        self.phone_number = _digits(phone_number)
        self.cooldown_seconds = cooldown_seconds
        self.opener = opener or webbrowser.open
        self.dry_run = dry_run
        # Base URL of the evidence server (no trailing slash), '' when off.
        self.evidence_base_url = (evidence_base_url or "").rstrip("/")
        self.last_escalation_time = 0

    @property
    def configured(self):
        return bool(self.whatsapp_number or self.phone_number)

    def _evidence_url(self, path):
        """Photo URL when the evidence server is up, else '' (caller falls back)."""
        if not path or not self.evidence_base_url:
            return ""
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        return f"{self.evidence_base_url}/{name}"

    def build_message(self, local_decision, model_decision, evidence=None,
                      alert_message=""):
        evidence = evidence or {}
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "[Crowd Safety] Designated-reviewer notification (not an emergency service).",
            f"Time: {timestamp}",
            f"Local verdict: risk={local_decision.get('risk_level')} action={local_decision.get('action')}",
            f"Local reasoning: {local_decision.get('reasoning', '')}",
        ]
        if isinstance(model_decision, dict):
            lines.append(
                f"Model verdict: risk={model_decision.get('risk_level')} action={model_decision.get('action')}"
            )
            lines.append(f"Model reasoning: {model_decision.get('reasoning', '')}")
        else:
            lines.append("Model verdict: unavailable (call failed or invalid output)")

        # The decision agent writes this; it is the narrative the human reads
        # first. Falls back to the structured lines above when unavailable.
        alert_message = (alert_message or "").strip()
        if alert_message:
            lines.append("")
            lines.append(alert_message)
            lines.append("")

        photo_url = self._evidence_url(evidence.get("frame_path"))
        if photo_url:
            lines.append(f"Snapshot: {photo_url}")
        elif evidence.get("frame_path"):
            lines.append(f"Screenshot: {evidence['frame_path']}")
        if evidence.get("clip_path"):
            lines.append(f"Clip: {evidence['clip_path']}")
        lines.append("Please review the evidence and respond.")
        return "\n".join(lines)

    def build_links(self, message):
        links = []
        if self.whatsapp_number:
            links.append(build_wa_link(self.whatsapp_number, message))
        if self.phone_number:
            links.append(f"tel:{self.phone_number}")
        return links

    def escalate(self, local_decision, model_decision, evidence=None,
                 alert_message=""):
        """Opens the reviewer's wa.me + tel: links. Returns True if sent."""
        if not self.configured:
            print("[WARN] Escalation skipped: authority numbers are not configured")
            return False

        now = time.time()
        if now - self.last_escalation_time < self.cooldown_seconds:
            print("[WARN] Escalation skipped: cooldown active")
            return False

        message = self.build_message(
            local_decision, model_decision, evidence, alert_message=alert_message
        )
        for link in self.build_links(message):
            if self.dry_run:
                print(f"[ESCALATION DRY-RUN] {link}")
            else:
                self.opener(link)

        self.last_escalation_time = now
        return True


class ReviewerNotifier:
    """Sends the human-evaluation request to the reviewer's wa.me chat.

    Send-only: the approve/reject command comes back over the console prompt,
    because wa.me links cannot receive replies.
    """

    def __init__(self, whatsapp_number="", cooldown_seconds=60, opener=None,
                 dry_run=False, evidence_base_url=""):
        self.whatsapp_number = _digits(whatsapp_number)
        self.cooldown_seconds = cooldown_seconds
        self.opener = opener or webbrowser.open
        self.dry_run = dry_run
        self.evidence_base_url = (evidence_base_url or "").rstrip("/")
        self.last_request_time = 0

    @property
    def configured(self):
        return bool(self.whatsapp_number)

    def build_message(self, local_decision, verify_result, evidence=None,
                      alert_message=""):
        evidence = evidence or {}
        verify_result = verify_result or {}
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "[Crowd Safety] HUMAN EVALUATION REQUESTED (not an emergency service).",
            f"Time: {timestamp}",
            f"Reason: {verify_result.get('reason', 'verification unavailable')}",
            f"Local verdict: risk={local_decision.get('risk_level')} action={local_decision.get('action')}",
        ]
        model_decision = verify_result.get("model_decision")
        if isinstance(model_decision, dict):
            lines.append(
                f"Verifier verdict: risk={model_decision.get('risk_level')} "
                f"action={model_decision.get('action')} | {model_decision.get('reasoning', '')}"
            )
        else:
            lines.append("Verifier verdict: unavailable (call failed or invalid output)")

        alert_message = (alert_message or verify_result.get("alert_message") or "").strip()
        if alert_message:
            lines.append(f"What the verifier saw: {alert_message}")

        frame_path = evidence.get("frame_path", "")
        if frame_path:
            if self.evidence_base_url:
                name = str(frame_path).replace("\\", "/").rsplit("/", 1)[-1]
                lines.append(f"Snapshot: {self.evidence_base_url}/{name}")
            else:
                lines.append(f"Screenshot: {frame_path}")

        lines.append(
            "Open the program console and type 'approve' to proceed "
            "or 'reject' to skip. Reply here if you cannot reach the console."
        )
        return "\n".join(lines)

    def request_review(self, local_decision, verify_result, evidence=None):
        """Opens the reviewer's wa.me link. Returns True if sent."""
        if not self.configured:
            return False

        now = time.time()
        if now - self.last_request_time < self.cooldown_seconds:
            print("[WARN] Review request skipped: cooldown active")
            return False

        link = build_wa_link(self.whatsapp_number, self.build_message(local_decision, verify_result, evidence))
        if self.dry_run:
            print(f"[REVIEW REQUEST DRY-RUN] {link}")
        else:
            self.opener(link)

        self.last_request_time = now
        return True
