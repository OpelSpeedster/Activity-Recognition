"""Non-blocking wrapper around a synchronous judge.

The video loop in main.py reads one frame at a time. A judge call can take
seconds, and blocking on it means frames stop being read and the webcam buffer
stalls. AsyncVerifier moves the call to a single worker thread so the loop can
keep running and pick the verdict up later.

Semantics are deliberately "latest wins": at most one job is running and one
is waiting, and a newer submission replaces a waiting one. Safety monitoring
cares about the current scene, not a backlog of stale frames, and this bounds
memory no matter how slow the judge is.

Each job carries the evidence captured when it was submitted, so escalation and
logging still reference the frame that was actually judged.

Completed verdicts buffer up to max_completed (default 8) instead of 1, because
the human-in-the-loop prompt can block the loop for a while and several verdicts
may land before they are drained. Silently dropping a finished high-risk verdict
would be worse than the memory it costs.
"""
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from agents.vlm_verifier import encode_frame_jpeg


class AsyncVerifier:
    """Runs a judge on one worker thread and hands results back on demand."""

    def __init__(self, verifier, max_completed=8):
        self.verifier = verifier
        self._lock = threading.Lock()
        self._busy = False
        self._pending = None
        self._completed = deque(maxlen=max_completed)
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge")

    @property
    def in_flight(self):
        """True while a verdict is still owed for some submitted job."""
        with self._lock:
            return self._busy or self._pending is not None

    def submit(self, frame_id, frame, context, local_decision, evidence=None):
        """Queues a verification. Returns True if the worker took it now.

        The frame is encoded on the calling thread so the numpy array is never
        shared across threads.
        """
        job = {
            "frame_id": frame_id,
            "context": context,
            "decision": local_decision,
            "evidence": dict(evidence or {}),
            "_b64": encode_frame_jpeg(frame),
        }
        with self._lock:
            if self._closed:
                return False
            if self._busy:
                # Supersede: only the newest unstarted frame is worth judging.
                self._pending = job
                return False
            self._busy = True
        self._executor.submit(self._run, job)
        return True

    def _run(self, job):
        try:
            job["result"] = self.verifier.verify_encoded(
                job.pop("_b64"), job["context"], job["decision"]
            )
        except Exception as exc:
            # The judges already degrade internally; this is a last resort so a
            # worker thread can never die silently and strand the pipeline.
            job["result"] = {
                "agreement": "insufficient_output",
                "model_decision": None,
                "reason": f"AI judge raised unexpectedly: {exc}",
                "latency_ms": 0.0,
                "model": getattr(self.verifier, "model", ""),
            }
        with self._lock:
            self._completed.append(job)
            self._busy = False
            next_job = self._pending
            self._pending = None
            if next_job is not None and not self._closed:
                self._busy = True
                self._executor.submit(self._run, next_job)

    def poll(self):
        """Returns the oldest finished job, or None. Never blocks.

        A job is {frame_id, context, local_decision, evidence, result}.
        """
        with self._lock:
            if not self._completed:
                return None
            return self._completed.popleft()

    def shutdown(self, wait=True):
        """Stops accepting work. Waits for an in-flight judge call if asked."""
        with self._lock:
            self._closed = True
            self._pending = None
        self._executor.shutdown(wait=wait)
