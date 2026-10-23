from collections import deque
from datetime import datetime
from pathlib import Path

import cv2


class ClipBuffer:
    def __init__(self, fps, seconds, output_dir):
        self.fps = max(fps, 1)
        self.max_len = int(self.fps * seconds)
        self.frames = deque(maxlen=self.max_len)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, frame):
        self.frames.append(frame.copy())

    def save_clip(self, tag="incident"):
        if len(self.frames) < 2:
            return ""

        height, width = self.frames[0].shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{tag}_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (width, height))
        for frame in self.frames:
            writer.write(frame)
        writer.release()

        return str(path)
