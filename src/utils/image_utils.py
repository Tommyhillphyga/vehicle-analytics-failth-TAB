from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def clip_box(box: list[float] | tuple[float, ...], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(value) for value in box[:4])
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def crop_image(image: np.ndarray, box: list[float] | tuple[float, ...]) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None
    x1, y1, x2, y2 = clip_box(box, image.shape[1], image.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def save_image(image: np.ndarray | None, path: str | Path) -> bool:
    if image is None or image.size == 0:
        return False
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(destination), image))