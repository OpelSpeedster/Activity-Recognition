"""
Thin wrapper around Ultralytics YOLOv8-Pose.

Returns:
    detections: list of {"bbox": [x1,y1,x2,y2], "confidence": float, "class_id": 0}
    keypoints_list: list of 17 (x, y, confidence) tuples per detection, in
                     the same order as `detections` (COCO keypoint layout).
"""
from ultralytics import YOLO


class PoseEstimator:
    def __init__(self, model_path, conf=0.10, iou=0.65, device=None, imgsz=640, max_det=1000):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.max_det = max_det

    def estimate(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            classes=[0],
            verbose=False,
            device=self.device,
        )

        if not results:
            return [], []

        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return [], []

        keypoints_xy = result.keypoints.xy
        keypoints_conf = result.keypoints.conf
        detections = []
        keypoints_list = []

        for idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(box.conf[0]),
                    "class_id": 0,
                }
            )

            if keypoints_xy is None:
                keypoints_list.append([])
                continue

            points = []
            xy_points = keypoints_xy[idx].tolist()
            conf_points = keypoints_conf[idx].tolist() if keypoints_conf is not None else None
            for kp_idx, (x, y) in enumerate(xy_points):
                conf = float(conf_points[kp_idx]) if conf_points else 1.0
                points.append((x, y, conf))
            keypoints_list.append(points)

        return detections, keypoints_list
