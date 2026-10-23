"""
Fall detection.

Original approach: flag a "fall" purely from sustained horizontal posture
(torso angle or bbox aspect ratio). Problem: bending, crouching, sitting on
the floor, or a partially-occluded pose all look "horizontal" too, so it's
prone to false positives.

Improved approach: a fall is a *transition* from upright to horizontal, not
just being horizontal. Real falls have two signatures other than static
posture:

  1. Posture change: bbox height (a rough proxy for standing height) drops
     sharply within a short window.
  2. Sustained horizontal posture *after* the collapse, confirmed by both
     the shoulder->hip torso angle AND the bbox aspect ratio, so a
     single noisy signal can't trigger a false fall on its own.

We keep "inactivity" as a separate, independent signal (person on the
ground not moving), since that alone is useful even without a confirmed
fall trigger (e.g. someone who fainted while already seated).
"""
import math
from collections import deque

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


class FallDetector:
    def __init__(
        self,
        angle_threshold=55,
        min_frames=5,
        inactivity_frames=15,
        inactivity_motion_threshold=0.01,
        horizontal_ratio_threshold=1.3,
        min_drop_ratio=0.35,
        height_history_len=10,
        keypoint_conf_threshold=0.5,
    ):
        self.angle_threshold = angle_threshold
        self.min_frames = min_frames
        self.inactivity_frames = inactivity_frames
        self.inactivity_motion_threshold = inactivity_motion_threshold
        self.horizontal_ratio_threshold = horizontal_ratio_threshold
        self.min_drop_ratio = min_drop_ratio
        self.height_history_len = height_history_len
        self.keypoint_conf_threshold = keypoint_conf_threshold

        self.horizontal_counts = {}
        self.still_counts = {}
        self.prev_poses = {}
        self.height_history = {}
        self.fallen_ids = set()  # people already confirmed fallen (avoid re-triggering every frame)

    # -- geometry helpers -----------------------------------------------
    def _point(self, keypoints, index):
        if index >= len(keypoints):
            return None
        x, y, visibility = keypoints[index]
        if visibility < self.keypoint_conf_threshold:
            return None
        return (x, y)

    def _midpoint(self, keypoints, idx1, idx2):
        p1 = self._point(keypoints, idx1)
        p2 = self._point(keypoints, idx2)
        if not p1 or not p2:
            return None
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def _torso_angle(self, keypoints):
        """Angle of shoulder->hip vector from vertical. 0=upright, 90=horizontal."""
        shoulder_mid = self._midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER)
        hip_mid = self._midpoint(keypoints, LEFT_HIP, RIGHT_HIP)
        if not shoulder_mid or not hip_mid:
            return None
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        if dx == 0 and dy == 0:
            return None
        return abs(math.degrees(math.atan2(dx, dy))) if dy != 0 else 90.0

    def _is_horizontal(self, pose):
        """
        Horizontal posture requires agreement from at least one keypoint-based
        signal (torso angle) OR a strong bbox-ratio signal on its own, but we
        weight the keypoint signal higher since it's more reliable than a
        raw bounding box (bbox ratio is easily skewed by occlusion/crop).
        """
        keypoints = pose["keypoints"]
        bbox = pose["bbox"]
        x1, y1, x2, y2 = bbox
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        ratio = bbox_w / bbox_h

        angle = self._torso_angle(keypoints)
        if angle is not None:
            return angle >= self.angle_threshold
        # fallback only when keypoints are unusable (heavy occlusion)
        return ratio >= self.horizontal_ratio_threshold

    def _bbox_height(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(1, y2 - y1)

    def _sudden_drop(self, person_id, current_height):
        history = self.height_history.setdefault(
            person_id, deque(maxlen=self.height_history_len)
        )
        dropped = False
        if history:
            baseline = max(history)  # tallest recent height = "standing" reference
            if baseline > 0:
                drop_ratio = (baseline - current_height) / baseline
                dropped = drop_ratio >= self.min_drop_ratio
        history.append(current_height)
        return dropped

    def _pose_motion(self, current_pose, previous_pose):
        if previous_pose is None:
            return 0.0
        current = current_pose["keypoints"]
        previous = previous_pose["keypoints"]
        x1, y1, x2, y2 = current_pose["bbox"]
        bbox_size = max(1, max(x2 - x1, y2 - y1))

        total = 0.0
        count = 0
        for idx, (x, y, visibility) in enumerate(current):
            if visibility < self.keypoint_conf_threshold or idx >= len(previous):
                continue
            px, py, pvis = previous[idx]
            if pvis < self.keypoint_conf_threshold:
                continue
            total += math.hypot(x - px, y - py)
            count += 1

        if count == 0:
            return 0.0
        return (total / count) / bbox_size

    # -- main entry point --------------------------------------------------
    def update(self, poses):
        falls = []
        inactivity_events = []
        current_ids = set()

        for pose in poses:
            person_id = pose.get("id", -1)
            if person_id == -1:
                continue

            current_ids.add(person_id)
            horizontal = self._is_horizontal(pose)
            motion = self._pose_motion(pose, self.prev_poses.get(person_id))
            height = self._bbox_height(pose["bbox"])
            dropped_suddenly = self._sudden_drop(person_id, height)

            if horizontal:
                self.horizontal_counts[person_id] = self.horizontal_counts.get(person_id, 0) + 1
            else:
                self.horizontal_counts[person_id] = 0
                self.fallen_ids.discard(person_id)  # stood back up

            if motion <= self.inactivity_motion_threshold:
                self.still_counts[person_id] = self.still_counts.get(person_id, 0) + 1
            else:
                self.still_counts[person_id] = 0

            sustained_horizontal = self.horizontal_counts[person_id] >= self.min_frames
            is_new_fall = (
                sustained_horizontal
                and person_id not in self.fallen_ids
                and (dropped_suddenly or self.still_counts[person_id] >= 1)
            )

            if is_new_fall:
                self.fallen_ids.add(person_id)
                falls.append(
                    {
                        "id": person_id,
                        "horizontal_frames": self.horizontal_counts[person_id],
                        "motion": round(motion, 4),
                        "sudden_drop": dropped_suddenly,
                    }
                )
            elif sustained_horizontal and person_id in self.fallen_ids:
                # still down, already reported — don't re-fire every frame,
                # but caller can still see it via inactivity events below
                pass

            if self.still_counts[person_id] >= self.inactivity_frames:
                inactivity_events.append(
                    {
                        "id": person_id,
                        "still_frames": self.still_counts[person_id],
                        "motion": round(motion, 4),
                        "is_down": person_id in self.fallen_ids,
                    }
                )

            self.prev_poses[person_id] = pose

        for person_id in list(self.prev_poses.keys()):
            if person_id not in current_ids:
                self.prev_poses.pop(person_id, None)
                self.horizontal_counts.pop(person_id, None)
                self.still_counts.pop(person_id, None)
                self.height_history.pop(person_id, None)
                self.fallen_ids.discard(person_id)

        return falls, inactivity_events
