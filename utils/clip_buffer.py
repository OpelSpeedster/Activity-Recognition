"""Rolling video history and event-centered clip capture."""
from collections import deque
from datetime import datetime
from pathlib import Path
import re

import cv2


def sample_event_frames(frames, trigger_index, pre_count, post_count):
    """Return frames sampled before, at, and after the event trigger."""
    before_index = trigger_index - pre_count
    after_index = trigger_index + post_count
    if before_index < 0 or trigger_index < 0 or after_index >= len(frames):
        raise ValueError("event clip is missing a temporal phase")
    return [frames[before_index], frames[trigger_index], frames[after_index]]


class ClipBuffer:
    """Keep a rolling pre-roll and finalize one clip per triggered event."""

    def __init__(self, fps, seconds, output_dir):
        self.fps = max(float(fps), 1.0)
        self.seconds = max(float(seconds), 1.0 / self.fps)
        self.max_len = max(1, int(round(self.fps * self.seconds)))
        self.pre_count = max(0, round(self.fps * self.seconds / 2))
        self.post_count = self.pre_count
        self.frames = deque(maxlen=self.max_len)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pending = {}

    def add(self, frame):
        """Store a frame and return any event clips that reached post-roll."""
        stored_frame = frame.copy()
        self.frames.append(stored_frame)
        completed = []
        for event_id, event in list(self._pending.items()):
            event["frames"].append(stored_frame)
            event["remaining"] -= 1
            if event["remaining"] <= 0:
                completed.append(self._finish_event(event_id))
        return completed

    def trigger(self, event_id, metadata):
        """Start a clip from the rolling pre-roll through the configured post-roll."""
        event_id = str(event_id)
        if event_id in self._pending or not self.frames:
            return False
        event_frames = list(self.frames)
        self._pending[event_id] = {
            "event_id": event_id,
            "metadata": dict(metadata or {}),
            "frames": event_frames,
            "trigger_index": len(event_frames) - 1,
            "remaining": self.post_count,
        }
        if self.post_count == 0:
            self._finish_event(event_id)
        return True

    def flush_pending(self):
        """Finalize clips at end-of-input without fabricating post-roll frames."""
        return [self._finish_event(event_id) for event_id in list(self._pending)]

    def _finish_event(self, event_id):
        event = self._pending.pop(event_id)
        clip_path = self._write_event_clip(event["frames"], event_id)
        return {
            "event_id": event_id,
            "clip_path": clip_path,
            "frames": event["frames"],
            "trigger_index": event["trigger_index"],
            "metadata": event["metadata"],
        }

    def _write_event_clip(self, frames, event_id):
        if len(frames) < 2:
            return ""

        height, width = frames[0].shape[:2]
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(event_id)).strip("_") or "event"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"event_{safe_id}_{timestamp}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        try:
            if not writer.isOpened():
                return ""
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return str(path)

    def save_clip(self, tag="incident"):
        """Compatibility helper for the existing high-risk snapshot path."""
        if len(self.frames) < 2:
            return ""

        height, width = self.frames[0].shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{tag}_{timestamp}.mp4"

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        try:
            if not writer.isOpened():
                return ""
            for frame in self.frames:
                writer.write(frame)
        finally:
            writer.release()

        return str(path)