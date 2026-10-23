"""
Fight / aggressive-motion detection.

Original approach: flag a fight the instant avg or spike motion crosses a
threshold AND two people are near each other. Problem: a single frame of
motion blur, a high-five, a quick hug, or a tracking ID glitch can trigger
a one-frame false positive.

Improvement: require the near+motion condition to hold for
`min_consec_frames` in a row before it counts as a fight event. This is a
simple temporal-smoothing / debounce technique — trades a small amount of
detection latency (a few frames, <200ms at 30fps) for a large drop in
false-positive rate.
"""
import math
from collections import deque


class FightDetector:
    def __init__(
        self,
        motion_threshold=0.035,
        spike_threshold=0.06,
        near_distance_ratio=0.12,
        history_len=5,
        min_consec_frames=3,
        keypoint_conf_threshold=0.5,
    ):
        self.motion_threshold = motion_threshold
        self.spike_threshold = spike_threshold
        self.near_distance_ratio = near_distance_ratio
        self.history_len = history_len
        self.min_consec_frames = min_consec_frames
        self.keypoint_conf_threshold = keypoint_conf_threshold

        self.prev_poses = {}
        self.motion_history = {}
        self.consec_flag_counts = {}
        self.reported_ids = set()

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

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

    def update(self, poses, frame_shape):
        fights = []
        current_ids = set()
        pose_map = {}
        centers = {}

        for pose in poses:
            person_id = pose.get("id", -1)
            if person_id == -1:
                continue
            current_ids.add(person_id)
            pose_map[person_id] = pose
            centers[person_id] = self._center(pose["bbox"])

        near_ids = set()
        height, width = frame_shape[:2]
        near_distance = self.near_distance_ratio * max(width, height)
        ids = list(centers.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dist = math.hypot(
                    centers[ids[i]][0] - centers[ids[j]][0],
                    centers[ids[i]][1] - centers[ids[j]][1],
                )
                if dist <= near_distance:
                    near_ids.add(ids[i])
                    near_ids.add(ids[j])

        for person_id, pose in pose_map.items():
            motion = self._pose_motion(pose, self.prev_poses.get(person_id))
            history = self.motion_history.setdefault(person_id, deque(maxlen=self.history_len))
            history.append(motion)
            avg_motion = sum(history) / len(history)

            frame_flagged = (
                avg_motion >= self.motion_threshold or motion >= self.spike_threshold
            ) and person_id in near_ids

            if frame_flagged:
                self.consec_flag_counts[person_id] = self.consec_flag_counts.get(person_id, 0) + 1
            else:
                self.consec_flag_counts[person_id] = 0
                self.reported_ids.discard(person_id)

            if (
                self.consec_flag_counts[person_id] >= self.min_consec_frames
                and person_id not in self.reported_ids
            ):
                self.reported_ids.add(person_id)
                fights.append(
                    {
                        "id": person_id,
                        "motion": round(motion, 4),
                        "avg_motion": round(avg_motion, 4),
                        "consec_frames": self.consec_flag_counts[person_id],
                    }
                )

            self.prev_poses[person_id] = pose

        for person_id in list(self.prev_poses.keys()):
            if person_id not in current_ids:
                self.prev_poses.pop(person_id, None)
                self.motion_history.pop(person_id, None)
                self.consec_flag_counts.pop(person_id, None)
                self.reported_ids.discard(person_id)

        return fights
