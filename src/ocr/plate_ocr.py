from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

from src.utils.plate_utils import normalize_plate_text


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float


class PlateOCR:
    def __init__(self, minimum_confidence: float = 0.5, language: str = "en"):
        try:
            import easyocr
        except ImportError as error:
            raise RuntimeError("Install easyocr to use OCR.") from error
        self.minimum_confidence = minimum_confidence
        self.engine = easyocr.Reader([language], gpu=False)

    @staticmethod
    def preprocess(plate_image: Any) -> Any:
        resized = cv2.resize(plate_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def read(self, plate_image: Any) -> OCRResult | None:
        if plate_image is None or getattr(plate_image, "size", 0) == 0:
            return None
        image = self.preprocess(plate_image)
        candidates = []
        for _box, text, score in self.engine.readtext(image):
            confidence = float(score)
            normalized = str(text).upper()
            candidates.append(OCRResult(normalized, confidence))
        return max(candidates, key=lambda item: item.confidence, default=None)