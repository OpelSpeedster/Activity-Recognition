import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.telegram_notifier import TelegramNotifier


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class TestTelegramNotifier(unittest.TestCase):
    def test_sends_video_caption_with_alert_fields(self):
        requests = []

        def opener(request, timeout):
            requests.append(request)
            return FakeResponse({"ok": True})

        with TemporaryDirectory() as directory:
            clip_path = Path(directory) / "event.mp4"
            clip_path.write_bytes(b"video-content")
            notifier = TelegramNotifier("token", "123", opener=opener)
            result = notifier.send_event(
                {"alert_type": "possible_fall", "risk_level": "high", "summary": "Person on ground."},
                str(clip_path),
                "2026-10-27 10:00:00",
            )

        self.assertTrue(result["video_sent"])
        self.assertIn("sendVideo", requests[0].full_url)
        self.assertIn(b'name="video"', requests[0].data)
        self.assertIn(b"possible_fall", requests[0].data)
        self.assertIn(b"Person on ground.", requests[0].data)

    def test_missing_credentials_disables_delivery(self):
        self.assertFalse(TelegramNotifier("", "123").configured)

    def test_video_failure_uses_text_fallback(self):
        replies = iter([FakeResponse({"ok": False, "description": "video rejected"}), FakeResponse({"ok": True})])
        requests = []
        notifier = TelegramNotifier(
            "token", "123",
            opener=lambda request, timeout: (requests.append(request), next(replies))[1],
        )
        with TemporaryDirectory() as directory:
            clip_path = Path(directory) / "event.mp4"
            clip_path.write_bytes(b"video-content")
            result = notifier.send_event(
                {"alert_type": "inactivity", "risk_level": "medium", "summary": "Review needed."},
                str(clip_path),
                "2026-10-27 10:00:00",
            )

        self.assertTrue(result["text_sent"])
        self.assertEqual(len(requests), 2)
        self.assertIn("sendMessage", requests[1].full_url)
        self.assertIn(b"Clip upload failed", requests[1].data)

    def test_caption_is_limited_to_telegram_length(self):
        caption = TelegramNotifier.build_caption(
            {"alert_type": "uncertain", "risk_level": "medium", "summary": "🚨" * 2000},
            "now",
        )
        self.assertLessEqual(TelegramNotifier._utf16_length(caption), 1024)
        self.assertIn("Alert type: uncertain", caption)
        self.assertIn("Risk: medium", caption)

    def test_safe_error_redacts_bot_token(self):
        notifier = TelegramNotifier("secret-token", "123")
        self.assertNotIn("secret-token", notifier._safe_error(RuntimeError("bad secret-token URL")))


if __name__ == "__main__":
    unittest.main()