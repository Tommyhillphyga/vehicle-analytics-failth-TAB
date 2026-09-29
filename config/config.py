from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    vehicle_model_path: str = "models/yolov8m.pt"
    plate_model_path: str = "models/license_plate_detector.pt"
    vehicle_confidence: float = 0.40
    plate_confidence: float = 0.40
    ocr_confidence: float = 0.50
    iou_threshold: float = 0.50
    process_every_n_frames: int = 1
    database_path: str = "data/vehicle_analytics.db"
    vehicle_image_dir: str = "data/vehicles"
    plate_image_dir: str = "data/plates"
    video_dir: str = "data/videos"
    output_dir: str = "data/outputs"
    vehicle_classes: tuple[str, ...] = field(default=("car", "bus", "truck"))

    def ensure_directories(self) -> None:
        for directory in (
            self.database_path,
            self.vehicle_image_dir,
            self.plate_image_dir,
            self.video_dir,
            self.output_dir,
        ):
            Path(directory).parent.mkdir(parents=True, exist_ok=True) if Path(directory).suffix else Path(directory).mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = Settings()