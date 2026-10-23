"""
Minimal nearest-centroid tracker (greedy matching by distance).

Not a Kalman filter / Hungarian-algorithm tracker like DeepSORT or
ByteTrack -- it's intentionally simple: match each new detection to the
closest existing track center within `max_distance`, age out tracks that
go unmatched for `max_lost` frames. Good enough for a handful of people in
frame; will start mis-swapping IDs under heavy occlusion or fast crossing
paths with many people, because it's O(n*m) greedy nearest-neighbor, not a
globally optimal assignment (Hungarian algorithm) — two people crossing
paths can get their IDs swapped if their detections happen to be closer to
each other's previous track than their own.
"""
import math


class SimpleTracker:
    def __init__(self, max_lost=30, max_distance_ratio=0.1):
        self.max_lost = max_lost
        self.max_distance_ratio = max_distance_ratio
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, detections, frame_shape):
        height, width = frame_shape[:2]
        max_distance = self.max_distance_ratio * max(width, height)

        unmatched_tracks = set(self.tracks.keys())
        for det in detections:
            center = self._center(det["bbox"])
            best_id = None
            best_dist = None

            for track_id in list(unmatched_tracks):
                track = self.tracks[track_id]
                dist = math.hypot(center[0] - track["center"][0], center[1] - track["center"][1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            if best_id is not None and best_dist is not None and best_dist <= max_distance:
                det["id"] = best_id
                self.tracks[best_id]["center"] = center
                self.tracks[best_id]["bbox"] = det["bbox"]
                self.tracks[best_id]["lost"] = 0
                unmatched_tracks.discard(best_id)
            else:
                det["id"] = self.next_id
                self.tracks[self.next_id] = {"center": center, "bbox": det["bbox"], "lost": 0}
                self.next_id += 1

        for track_id in list(unmatched_tracks):
            self.tracks[track_id]["lost"] += 1
            if self.tracks[track_id]["lost"] > self.max_lost:
                del self.tracks[track_id]

        return detections
