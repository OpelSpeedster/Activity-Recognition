"""Optional LAN static server so wa.me messages can carry the photo.

WhatsApp links are text-only, so an escalation message can only carry a URL.
This serves the project's frame directory over plain HTTP on your LAN so the
authority taps the link and sees the snapshot instead of a bare file path.

This is NOT a public server: it binds to your machine and only exposes files
under the configured evidence directory. Disabled by default
(EVIDENCE_SERVER_ENABLED=0).
"""
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _EvidenceHandler(SimpleHTTPRequestHandler):
    """Static handler that only ever serves files inside the evidence dir."""

    def log_message(self, format, *args):
        # Keep the video loop's console clean.
        pass

    def end_headers(self):
        # Evidence must not be cached: each escalation references a new frame.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class EvidenceServer:
    def __init__(self, directory, port=8090, public_host=""):
        self.directory = Path(directory).resolve()
        self.port = port
        self.public_host = public_host
        self._server = None
        self._thread = None

    @property
    def running(self):
        return self._server is not None

    @property
    def bound_port(self):
        """Actual listening port (useful when constructed with port=0)."""
        if self._server is None:
            return self.port
        return self._server.server_address[1]

    def start(self):
        """Starts the server in a daemon thread. Returns True when running."""
        if self.running:
            return True
        if not self.directory.is_dir():
            print(f"[WARN] Evidence dir not found: {self.directory}")
            return False

        handler = partial(_EvidenceHandler, directory=str(self.directory))
        try:
            # Bind to all interfaces so a phone on the LAN can reach it.
            self._server = ThreadingHTTPServer(("0.0.0.0", self.port), handler)
        except OSError as exc:
            print(f"[WARN] Evidence server could not start on port {self.port}: {exc}")
            return False

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(
            f"[INFO] Evidence server on port {self.bound_port} serving {self.directory} "
            f"({'http://' + self.public_host + ':' + str(self.bound_port) if self.public_host else 'host not set'})"
        )
        return True

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None

    def url_for(self, relative_path):
        """Builds a LAN URL for a file under the evidence directory.

        relative_path may be absolute (as EventLogger returns) or already
        relative. Returns '' when the host is unset or the file is outside
        the evidence directory (never leak arbitrary paths over HTTP).
        """
        if not self.running or not self.public_host:
            return ""

        candidate = Path(relative_path)
        if not candidate.is_absolute():
            candidate = self.directory / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            return ""

        if self.directory not in resolved.parents and resolved != self.directory:
            return ""

        # Preserve the path relative to the served root (files are flat in
        # practice, but a nested one must still resolve to its real URL).
        try:
            relative = resolved.relative_to(self.directory)
        except ValueError:
            return ""
        if relative == Path("."):
            return ""
        return f"http://{self.public_host}:{self.bound_port}/{relative.as_posix()}"
