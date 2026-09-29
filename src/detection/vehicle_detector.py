from typing import Any


COCO_VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


class VehicleDetector:
    def __init__(self, model_path: str, confidence: float = 0.4, iou: float = 0.5):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install ultralytics to use vehicle detection.") from error
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou

    def track(self, frame: Any) -> list[dict]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(COCO_VEHICLE_CLASSES),
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        detections = []
        for index, box in enumerate(boxes):
            class_id = int(box.cls.item())
            track_id = int(box.id.item()) if box.id is not None else None
            detections.append(
                {
                    "bbox": [float(value) for value in box.xyxy[0].tolist()],
                    "confidence": float(box.conf.item()),
                    "class_id": class_id,
                    "class_name": COCO_VEHICLE_CLASSES.get(class_id, "vehicle"),
                    "track_id": track_id,
                }
            )
        return detections 