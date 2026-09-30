from typing import Any


class PlateDetector:
    def __init__(self, model_path: str, confidence: float = 0.5):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install ultralytics to use plate detection.") from error
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: Any) -> list[dict]:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        if not results:
            return []
        detections = []
        for box in results[0].boxes:
            detections.append(
                {
                    "bbox": [float(value) for value in box.xyxy[0].tolist()],
                    "confidence": float(box.conf.item()),
                    "class_id": int(box.cls.item()),
                }
            )
        return detections