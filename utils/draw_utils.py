import cv2

# COCO keypoint skeleton for YOLOv8 pose (17 points).
POSE_CONNECTIONS = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        person_id = det.get("id", -1)
        conf = det.get("confidence", 0)
        label = f"ID {person_id} {conf:.2f}" if person_id != -1 else f"{conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def draw_poses(frame, poses):
    for pose in poses:
        keypoints = pose["keypoints"]
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue

            x1, y1, v1 = keypoints[start_idx]
            x2, y2, v2 = keypoints[end_idx]
            if v1 < 0.5 or v2 < 0.5:
                continue
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        for x, y, v in keypoints:
            if v < 0.5:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)


def draw_status(frame, lines, color=(0, 255, 255)):
    y = 25
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 22
