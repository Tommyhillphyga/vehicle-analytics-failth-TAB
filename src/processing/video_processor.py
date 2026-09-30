from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2

from config.config import Settings
from src.database.database import VehicleDatabase
from src.detection.plate_detector import PlateDetector
from src.detection.vehicle_detector import VehicleDetector
from src.ocr.plate_ocr import PlateOCR
from src.processing.annotations import annotate_frame
from src.tracking.vehicle_tracker import VehicleTracker
from src.utils.image_utils import clip_box, crop_image, save_image
from src.utils.plate_utils import safe_plate_filename


class VideoProcessor:
    def __init__(self, settings: Settings, database: VehicleDatabase, vehicle_detector: VehicleDetector, plate_detector: PlateDetector, ocr: PlateOCR):
        self.settings = settings
        self.database = database
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr = ocr

    def process(self, video_path: str, on_frame: Callable[[object, dict], None] | None = None) -> dict:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        tracker = VehicleTracker()
        detections = []
        frame_number = 0
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                current_frame = frame_number
                frame_number += 1
                if current_frame % self.settings.process_every_n_frames:
                    continue
                vehicles = self.vehicle_detector.track(frame)
                tracks = tracker.update(vehicles, current_frame)
                for track in tracks:
                    vehicle = next(item for item in vehicles if item.get("track_id") == track.track_id)
                    if track.vehicle_image_path is None:
                        crop = crop_image(frame, vehicle["bbox"])
                        path = Path(self.settings.vehicle_image_dir) / f"vehicle_{track.track_id:04d}.jpg"
                        if save_image(crop, path):
                            track.vehicle_image_path = str(path)
                    self.database.upsert_entry(self._entry(track, fps))

                plate_detections = []
                for vehicle in vehicles:
                    vehicle_crop = crop_image(frame, vehicle["bbox"])
                    if vehicle_crop is None:
                        continue
                    vehicle_x1, vehicle_y1, _, _ = clip_box(vehicle["bbox"], frame.shape[1], frame.shape[0])
                    track_id = vehicle.get("track_id")
                    track = tracker.tracks.get(track_id)

                    for plate in self.plate_detector.detect(vehicle_crop):
                        local_box = plate["bbox"]
                        plate["bbox"] = [
                            local_box[0] + vehicle_x1,
                            local_box[1] + vehicle_y1,
                            local_box[2] + vehicle_x1,
                            local_box[3] + vehicle_y1,
                        ]
                        plate_crop = crop_image(frame, plate["bbox"])
                        result = self.ocr.read(plate_crop)
                        if result is not None:
                            if track is not None:
                                track.plate_number = result.text
                                track.plate_confidence = result.confidence
                                path = Path(self.settings.plate_image_dir) / f"{safe_plate_filename(result.text)}_{track.track_id}.jpg"
                                if save_image(plate_crop, path):
                                    track.plate_image_path = str(path)
                                self.database.upsert_entry(self._entry(track, fps))
                        plate["text"] = result.text if result else None
                        plate["confidence"] = result.confidence if result else None
                        plate["track_id"] = track_id
                        detections.append({"frame": current_frame, **plate.copy()})
                        plate_detections.append(plate)

                annotated = annotate_frame(frame, vehicles, plate_detections)
                if on_frame:
                    on_frame(
                        annotated,
                        {
                            "frame": current_frame,
                            "total_frames": total_frames,
                            "total_vehicles": len(tracker.tracks),
                            "counts": tracker.counts_by_type(),
                        },
                    )
        finally:
            capture.release()
        return {
            "tracks": tracker.tracks,
            "counts": tracker.counts_by_type(),
            "frames": frame_number,
            "detections": detections,
        }

    @staticmethod
    def _entry(track, fps: float) -> dict:
        return {
            "track_id": track.track_id,
            "vehicle_type": track.vehicle_type,
            "plate_number": track.plate_number,
            "entry_time": datetime.fromtimestamp(track.first_seen_frame / fps).strftime("%H:%M:%S"),
            "vehicle_image_path": track.vehicle_image_path,
            "plate_image_path": track.plate_image_path,
        }