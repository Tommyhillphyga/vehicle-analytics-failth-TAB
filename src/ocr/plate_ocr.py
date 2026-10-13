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
    def __init__(self, minimum_confidence: float = 0.3, language: str = "en"):
        try:
            import easyocr
        except ImportError as error:
            raise RuntimeError("Install easyocr to use OCR.") from error
        self.minimum_confidence = minimum_confidence
        self.engine = easyocr.Reader([language], gpu=False)

    @staticmethod
    def preprocess(plate_image: Any) -> Any:
        h, w,_ = plate_image.shape
        y_start, y_end = int(h * 0.15), int(h * 0.90)
        x_start, x_end = int(w * 0.05), int(w * 0.95)
        cropped = plate_image[y_start:y_end, x_start:x_end]

        # resized = cv2.resize(plate_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cropped_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE to enhance local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cropped_enhanced = clahe.apply(cropped_gray)
      
        return cropped_enhanced

    def read(self, plate_image: Any) -> OCRResult | None:
        if plate_image is None or getattr(plate_image, "size", 0) == 0:
            return None
        image = self.preprocess(plate_image)
        candidates = []
        for _box, text, score in self.engine.readtext(image):
            confidence = float(f"{score:.4f}")
            if len(str(text)) < 9:  # Adjust the minimum length as needed
                continue
            normalized = str(text).upper()
            print(f"text:{normalized} length: {len(normalized)}, confidence: {confidence}")
            candidates.append(OCRResult(normalized, confidence))
        return max(candidates, key=lambda item: item.confidence, default=None)