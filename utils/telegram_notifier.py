"""Telegram Bot API delivery for human-reviewed event alerts."""
import json
from pathlib import Path
import uuid
from urllib.request import Request, urlopen


MAX_VIDEO_BYTES = 50_000_000
MAX_CAPTION_UTF16_UNITS = 1024
MAX_MESSAGE_CHARACTERS = 4096


class TelegramNotifier:
    def __init__(self, token, chat_id, timeout=10.0, opener=None):
        self.token = str(token or "").strip()
        self.chat_id = str(chat_id or "").strip()
        self.timeout = float(timeout)
        self.opener = opener or urlopen
        self.configured = bool(self.token and self.chat_id)

    @staticmethod
    def _utf16_length(value):
        return len(value.encode("utf-16-le")) // 2

    @classmethod
    def _truncate_utf16(cls, value, limit):
        result = []
        used = 0
        for char in value:
            units = cls._utf16_length(char)
            if used + units > limit:
                break
            result.append(char)
            used += units
        return "".join(result)

    @classmethod
    def build_caption(cls, decision, timestamp):
        alert_type = str((decision or {}).get("alert_type") or "uncertain").strip()
        risk_level = str((decision or {}).get("risk_level") or "medium").strip()
        summary = str((decision or {}).get("summary") or "No summary provided.").strip()
        prefix = f"Alert type: {alert_type}\nRisk: {risk_level}\nTime: {timestamp}\nSummary: "
        available = max(0, MAX_CAPTION_UTF16_UNITS - cls._utf16_length(prefix))
        summary = cls._truncate_utf16(summary, available)
        return prefix + summary

    def send_event(self, decision, clip_path, timestamp):
        caption = self.build_caption(decision, timestamp)
        if not self.configured:
            return {"video_sent": False, "text_sent": False, "error": "Telegram is not configured"}

        try:
            self._post_multipart(
                "sendVideo",
                {"chat_id": self.chat_id, "caption": caption},
                clip_path,
            )
            return {"video_sent": True, "text_sent": False, "error": ""}
        except Exception as exc:
            safe_error = self._safe_error(exc)
            note = f"{caption}\nClip upload failed: {safe_error}"
            note = self._truncate_utf16(note, MAX_MESSAGE_CHARACTERS)
            try:
                self._post_json("sendMessage", {"chat_id": self.chat_id, "text": note})
                return {"video_sent": False, "text_sent": True, "error": safe_error}
            except Exception as text_exc:
                return {
                    "video_sent": False,
                    "text_sent": False,
                    "error": self._safe_error(text_exc),
                }

    def _api_url(self, method):
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def _open_json_response(self, request):
        with self.opener(request, timeout=self.timeout) as response:
            raw = response.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Telegram returned an invalid response") from exc
        if not isinstance(payload, dict) or not payload.get("ok"):
            description = payload.get("description", "Telegram API request failed") if isinstance(payload, dict) else "Telegram API request failed"
            raise RuntimeError(str(description))
        return payload

    def _post_json(self, method, payload):
        request = Request(
            self._api_url(method),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        return self._open_json_response(request)

    def _post_multipart(self, method, fields, clip_path):
        path = Path(str(clip_path or ""))
        if not path.is_file():
            raise FileNotFoundError("event clip is missing")
        size = path.stat().st_size
        if size > MAX_VIDEO_BYTES:
            raise ValueError("event clip exceeds Telegram's 50 MB upload limit")

        boundary = f"----CodexTelegram{uuid.uuid4().hex}"
        chunks = []
        for name, value in fields.items():
            chunks.extend([
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
                str(value).encode("utf-8"),
                b"\r\n",
            ])

        filename = path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        chunks.extend([
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="video"; filename="{filename}"\r\n'.encode("utf-8"),
            b"Content-Type: video/mp4\r\n\r\n",
            path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ])
        request = Request(
            self._api_url(method),
            data=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        return self._open_json_response(request)

    def _safe_error(self, exc):
        message = str(exc) or exc.__class__.__name__
        if self.token:
            message = message.replace(self.token, "[REDACTED]")
        return message[:1000]