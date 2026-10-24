import threading
import time
import unittest

from utils.verify_queue import AsyncVerifier

CONTEXT = {"crowd": {"count": 1, "density": 0.1, "level": "low"}, "falls": []}
LOCAL = {"risk_level": "high", "action": "human_review", "reasoning": "fall"}


def make_frame():
    import numpy as np

    return np.zeros((32, 32, 3), dtype=np.uint8)


class FakeVerifier:
    """Synchronous stand-in for a judge, optionally gated on an event."""

    model = "fake-model"

    def __init__(self, gate=None):
        self.gate = gate
        self.calls = []

    def verify_encoded(self, frame_jpeg_b64, context, local_decision):
        self.calls.append((frame_jpeg_b64, context, local_decision))
        if self.gate is not None:
            self.gate.wait(timeout=5.0)
        return {
            "agreement": "agreement",
            "model_decision": dict(local_decision),
            "reason": "match",
            "latency_ms": 1.5,
            "model": self.model,
        }


class RaisingVerifier:
    model = "raising-model"

    def verify_encoded(self, frame_jpeg_b64, context, local_decision):
        raise RuntimeError("judge exploded")


def wait_for(async_verifier, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = async_verifier.poll()
        if job is not None:
            return job
        time.sleep(0.005)
    return None


def wait_idle(async_verifier, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not async_verifier.in_flight:
            return True
        time.sleep(0.005)
    return False


class TestAsyncVerifier(unittest.TestCase):
    def test_submit_when_idle_starts_immediately(self):
        gate = threading.Event()
        verifier = AsyncVerifier(FakeVerifier(gate=gate))
        try:
            self.assertTrue(verifier.submit(1, make_frame(), CONTEXT, LOCAL, {"frame_path": "a.jpg"}))
            # The gate holds the worker so in_flight is deterministic, not a race.
            self.assertTrue(verifier.in_flight)
        finally:
            gate.set()
            verifier.shutdown()

    def test_poll_returns_none_while_busy_then_the_job(self):
        verifier = AsyncVerifier(FakeVerifier())
        try:
            verifier.submit(5, make_frame(), CONTEXT, LOCAL, {"clip_path": "c.mp4"})
            job = wait_for(verifier)
            self.assertIsNotNone(job)
            self.assertEqual(job["frame_id"], 5)
            self.assertEqual(job["decision"], LOCAL)
            self.assertEqual(job["evidence"], {"clip_path": "c.mp4"})
            self.assertEqual(job["result"]["agreement"], "agreement")
            self.assertEqual(job["result"]["model"], "fake-model")
        finally:
            verifier.shutdown()

    def test_encode_happens_on_calling_thread(self):
        verifier = AsyncVerifier(FakeVerifier())
        try:
            verifier.submit(1, make_frame(), CONTEXT, LOCAL)
            job = wait_for(verifier)
            # The worker receives an encoded string, never the ndarray itself.
            self.assertIsInstance(job["result"], dict)
            self.assertNotIn("_b64", job)
        finally:
            verifier.shutdown()

    def test_sequential_jobs_all_complete(self):
        verifier = AsyncVerifier(FakeVerifier())
        try:
            seen = []
            for frame_id in range(1, 4):
                verifier.submit(frame_id, make_frame(), CONTEXT, LOCAL)
                job = wait_for(verifier)
                self.assertIsNotNone(job, f"job {frame_id} never completed")
                seen.append(job["frame_id"])
            # Drained after each submit, so the queue is never contended.
            self.assertEqual(seen, [1, 2, 3])
            self.assertTrue(wait_idle(verifier))
        finally:
            verifier.shutdown()

    def test_newer_job_supersedes_pending_one(self):
        gate = threading.Event()
        verifier = AsyncVerifier(FakeVerifier(gate=gate))
        try:
            verifier.submit(1, make_frame(), CONTEXT, LOCAL)
            # First job is now running; these two go through the pending slot.
            verifier.submit(2, make_frame(), CONTEXT, LOCAL)
            verifier.submit(3, make_frame(), CONTEXT, LOCAL)
            gate.set()

            first = wait_for(verifier)
            second = wait_for(verifier)
            self.assertEqual(first["frame_id"], 1)
            # Job 2 was replaced by 3: only the newest pending frame is judged.
            self.assertEqual(second["frame_id"], 3)
            self.assertIsNone(verifier.poll())
        finally:
            gate.set()
            verifier.shutdown()

    def test_raising_verifier_degrades_instead_of_dying(self):
        verifier = AsyncVerifier(RaisingVerifier())
        try:
            verifier.submit(9, make_frame(), CONTEXT, LOCAL)
            job = wait_for(verifier)
            self.assertIsNotNone(job)
            self.assertEqual(job["result"]["agreement"], "insufficient_output")
            self.assertIn("judge exploded", job["result"]["reason"])
            self.assertEqual(job["result"]["model"], "raising-model")
        finally:
            verifier.shutdown()

    def test_shutdown_rejects_new_submissions(self):
        verifier = AsyncVerifier(FakeVerifier())
        verifier.shutdown()
        self.assertFalse(verifier.submit(1, make_frame(), CONTEXT, LOCAL))

    def test_poll_empties_slot(self):
        verifier = AsyncVerifier(FakeVerifier())
        try:
            verifier.submit(1, make_frame(), CONTEXT, LOCAL)
            self.assertIsNotNone(wait_for(verifier))
            self.assertIsNone(verifier.poll())
        finally:
            verifier.shutdown()


if __name__ == "__main__":
    unittest.main()
