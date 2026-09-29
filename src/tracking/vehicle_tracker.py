from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class VehicleTrack:
    track_id: int
    vehicle_type: str
    first_seen_frame: int
    last_seen_frame: int
    plate_number: str | None = None
    plate_confidence: float = 0.0
    vehicle_image_path: str | None = None
    plate_image_path: str | None = None


class VehicleTracker:
    def __init__(self):
        self.tracks: dict[int, VehicleTrack] = {}

    def update(self, detections: list[dict], frame_number: int) -> list[VehicleTrack]:
        active = []
        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            if track_id not in self.tracks:
                self.tracks[track_id] = VehicleTrack(
                    track_id=track_id,
                    vehicle_type=detection["class_name"],
                    first_seen_frame=frame_number,
                    last_seen_frame=frame_number,
                )
            track = self.tracks[track_id]
            track.last_seen_frame = frame_number
            active.append(track)
        return active

    def counts_by_type(self) -> dict[str, int]:
        return dict(Counter(track.vehicle_type for track in self.tracks.values()))