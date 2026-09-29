from pathlib import Path

import cv2
import numpy as np

from config.config import Settings
from src.database.database import VehicleDatabase
from src.ocr.plate_ocr import OCRResult, PlateOCR
from src.processing.video_processor import VideoProcessor
from src.tracking.vehicle_tracker import VehicleTracker
from src.utils.plate_utils import normalize_plate_text, safe_plate_filename


def test_plate_normalization_and_safe_filename():
    assert normalize_plate_text(" ab-123 xy ") == "AB123XY"
    assert safe_plate_filename("../../ab 123") == "AB123"


def test_plate_ocr_uses_highest_confidence_easyocr_result():
    class FakeReader:
        def readtext(self, _image):
            return [
                ([], "ab-123", 0.65),
                ([], "xy-987", 0.92),
                ([], "low-456", 0.3),
            ]

    ocr = PlateOCR.__new__(PlateOCR)
    ocr.minimum_confidence = 0.5
    ocr.engine = FakeReader()

    result = ocr.read(np.zeros((20, 60, 3), dtype=np.uint8))

    assert result == OCRResult("XY987", 0.92)


def test_tracker_counts_unique_ids():
    tracker = VehicleTracker()
    detections = [{"track_id": 7, "class_name": "car"}]
    tracker.update(detections, 1)
    tracker.update(detections, 2)
    assert tracker.counts_by_type() == {"car": 1}


def test_database_upsert_and_partial_search(tmp_path: Path):
    database = VehicleDatabase(str(tmp_path / "vehicles.db"))
    database.upsert_entry({
        "track_id": 7,
        "vehicle_type": "car",
        "plate_number": "AB123XY",
        "entry_time": "10:42:31",
        "vehicle_image_path": None,
        "plate_image_path": None,
    })
    database.upsert_entry({
        "track_id": 7,
        "vehicle_type": "car",
        "plate_number": "AB123XY",
        "entry_time": "10:42:31",
        "vehicle_image_path": "data/vehicles/vehicle_0007.jpg",
        "plate_image_path": None,
    })
    results = database.search("123")
    assert len(results) == 1
    assert results[0]["vehicle_image_path"].endswith("vehicle_0007.jpg")


def test_missing_plate_is_searchable_as_empty_result(tmp_path: Path):
    database = VehicleDatabase(str(tmp_path / "vehicles.db"))
    database.upsert_entry({
        "track_id": 1,
        "vehicle_type": "truck",
        "plate_number": None,
        "entry_time": "10:00:00",
        "vehicle_image_path": None,
        "plate_image_path": None,
    })
    assert database.search("ZZZ") == []


def test_video_processor_associates_plate_and_saves_crops(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (160, 100))
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    writer.write(frame)
    writer.write(frame)
    writer.release()

    class FakeVehicleDetector:
        def track(self, _frame):
            return [{"bbox": [10, 10, 130, 90], "class_name": "car", "track_id": 3, "confidence": 0.9}]

    class FakePlateDetector:
        def __init__(self):
            self.crop_shapes = []

        def detect(self, vehicle_crop):
            self.crop_shapes.append(vehicle_crop.shape[:2])
            return [{"bbox": [40, 45, 80, 60], "confidence": 0.9, "class_id": 0}]

    class FakeOCR:
        def __init__(self):
            self.crop_shapes = []

        def read(self, plate_crop):
            self.crop_shapes.append(plate_crop.shape)
            return OCRResult("AB123XY", 0.95) if len(self.crop_shapes) == 1 else None

    labels = []
    rectangles = []
    original_put_text = cv2.putText
    original_rectangle = cv2.rectangle

    def capture_label(image, text, *args, **kwargs):
        labels.append(text)
        return original_put_text(image, text, *args, **kwargs)

    def capture_rectangle(image, start, end, *args, **kwargs):
        rectangles.append((start, end))
        return original_rectangle(image, start, end, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_label)
    monkeypatch.setattr(cv2, "rectangle", capture_rectangle)

    settings = Settings(
        database_path=str(tmp_path / "vehicles.db"),
        vehicle_image_dir=str(tmp_path / "vehicles"),
        plate_image_dir=str(tmp_path / "plates"),
        process_every_n_frames=1,
    )
    database = VehicleDatabase(settings.database_path)
    annotated_frames = []
    plate_detector = FakePlateDetector()
    ocr = FakeOCR()
    result = VideoProcessor(settings, database, FakeVehicleDetector(), plate_detector, ocr).process(
        str(video_path),
        lambda frame, state: annotated_frames.append((frame, state)),
    )
    assert len(result["tracks"]) == 1
    assert "flow" not in result
    assert len(annotated_frames) == 2
    assert annotated_frames[0][0].shape == frame.shape
    assert "flow" not in annotated_frames[0][1]
    assert "AB123XY" in labels
    assert result["detections"][0]["text"] == "AB123XY"
    assert result["detections"][1]["text"] is None
    assert plate_detector.crop_shapes == [(80, 120), (80, 120)]
    assert ocr.crop_shapes == [(15, 40, 3), (15, 40, 3)]
    assert rectangles.count(((50, 55), (90, 70))) == 2
    assert labels.count("AB123XY") == 1
    assert database.search("AB123")[0]["track_id"] == 3
    assert Path(result["tracks"][3].vehicle_image_path).exists()
    assert Path(result["tracks"][3].plate_image_path).exists()