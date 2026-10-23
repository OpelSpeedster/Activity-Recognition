import cv2


class VideoSource:
    def __init__(self, source, is_webcam, target_fps=None):
        self.source = source
        self.is_webcam = is_webcam
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        if self.is_webcam and target_fps:
            self.cap.set(cv2.CAP_PROP_FPS, float(target_fps))

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30.0

    @classmethod
    def from_webcam(cls, index=0, target_fps=None):
        return cls(index, True, target_fps=target_fps)

    @classmethod
    def from_file(cls, path):
        return cls(path, False)

    def get_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
