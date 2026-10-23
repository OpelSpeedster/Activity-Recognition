"""
Crowd density estimation from person count relative to frame area.

`density` is not people-per-square-meter (we have no camera calibration),
it's a normalized people-per-area-unit score: count / (frame_area / area_scale).
Treat the medium/high thresholds as relative tuning knobs per-camera, not
universal constants — a wide top-down CCTV shot and a narrow doorway camera
need very different thresholds for the same real-world crowd density.
"""


class DensityEstimator:
    def __init__(self, area_scale=100000, medium_threshold=8, high_threshold=15):
        self.area_scale = area_scale
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold

    def estimate(self, detections, frame_shape):
        count = len(detections)
        height, width = frame_shape[:2]
        area = max(1, height * width)
        density = count / (area / self.area_scale)

        if density >= self.high_threshold:
            level = "high"
        elif density >= self.medium_threshold:
            level = "medium"
        else:
            level = "low"

        return {"count": count, "density": round(density, 4), "level": level}
