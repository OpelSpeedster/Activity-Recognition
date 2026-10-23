import csv
from datetime import datetime
from pathlib import Path

import cv2


class EventLogger:
    def __init__(self, log_dir, frame_dir):
        self.log_dir = Path(log_dir)
        self.frame_dir = Path(frame_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "events.csv"
        self._ensure_header()

    def _ensure_header(self):
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "event_type", "risk_level", "details", "frame_path"])

    def log_event(self, event_type, risk_level, details, frame=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frame_path = ""

        if frame is not None:
            name = f"{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = self.frame_dir / name
            cv2.imwrite(str(path), frame)
            frame_path = str(path)

        with open(self.log_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event_type, risk_level, details, frame_path])

        return frame_path
