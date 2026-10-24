import tempfile
import unittest
from pathlib import Path

from utils.evidence_server import EvidenceServer


class TestEvidenceServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "risk_high.jpg").write_bytes(b"\xff\xd8\xff")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "other.jpg").write_bytes(b"\xff\xd8\xff")
        # Loopback host + port=0 so the fetch test is reachable and parallel
        # runs never collide on a fixed port.
        self.server = EvidenceServer(self.tmp.name, port=0, public_host="127.0.0.1")
        self.assertTrue(self.server.start())

    def tearDown(self):
        self.server.stop()
        self.tmp.cleanup()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server.bound_port}"

    def test_serves_frame_over_http(self):
        import urllib.request

        url = self.server.url_for("risk_high.jpg")
        self.assertEqual(url, f"{self.base}/risk_high.jpg")
        with urllib.request.urlopen(url, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"\xff\xd8\xff")
            self.assertEqual(response.headers.get("Cache-Control"), "no-store")

    def test_url_for_absolute_path(self):
        absolute = str(self.root / "risk_high.jpg")
        self.assertEqual(self.server.url_for(absolute), f"{self.base}/risk_high.jpg")

    def test_url_for_preserves_relative_path(self):
        # A nested file keeps its path so the URL resolves to the real file.
        self.assertEqual(
            self.server.url_for(str(self.root / "sub" / "other.jpg")),
            f"{self.base}/sub/other.jpg",
        )

    def test_url_uses_configured_lan_host(self):
        lan = EvidenceServer(self.tmp.name, port=0, public_host="192.168.1.5")
        self.assertTrue(lan.start())
        try:
            self.assertEqual(
                lan.url_for("risk_high.jpg"),
                f"http://192.168.1.5:{lan.bound_port}/risk_high.jpg",
            )
        finally:
            lan.stop()

    def test_url_for_returns_empty_when_not_running(self):
        stopped = EvidenceServer(self.tmp.name, port=0, public_host="127.0.0.1")
        self.assertFalse(stopped.running)
        self.assertEqual(stopped.url_for("risk_high.jpg"), "")

    def test_url_for_returns_empty_without_public_host(self):
        hostless = EvidenceServer(self.tmp.name, port=0, public_host="")
        self.assertTrue(hostless.start())
        try:
            self.assertEqual(hostless.url_for("risk_high.jpg"), "")
        finally:
            hostless.stop()

    def test_url_for_refuses_path_outside_evidence_dir(self):
        # Path traversal above the served directory must never be published.
        self.assertEqual(self.server.url_for("..\\..\\secrets.txt"), "")
        self.assertEqual(self.server.url_for(str(self.root.parent / "secrets.txt")), "")

    def test_start_is_idempotent(self):
        first_port = self.server.bound_port
        self.assertTrue(self.server.start())
        self.assertEqual(self.server.bound_port, first_port)

    def test_stop_then_restart(self):
        self.server.stop()
        self.assertFalse(self.server.running)
        self.assertEqual(self.server.url_for("risk_high.jpg"), "")
        self.assertTrue(self.server.start())
        self.assertTrue(self.server.running)

    def test_start_fails_gracefully_for_missing_dir(self):
        server = EvidenceServer(str(self.root / "nope"), port=0)
        self.assertFalse(server.start())
        self.assertFalse(server.running)


if __name__ == "__main__":
    unittest.main()
