import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class VehicleDatabase:
    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS vehicle_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL UNIQUE,
                    vehicle_type TEXT NOT NULL,
                    plate_number TEXT,
                    entry_time TEXT NOT NULL,
                    vehicle_image_path TEXT,
                    plate_image_path TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def upsert_entry(self, entry: dict) -> None:
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO vehicle_entries
                (track_id, vehicle_type, plate_number, entry_time, vehicle_image_path, plate_image_path)
                VALUES (:track_id, :vehicle_type, :plate_number, :entry_time, :vehicle_image_path, :plate_image_path)
                ON CONFLICT(track_id) DO UPDATE SET
                    plate_number=COALESCE(excluded.plate_number, vehicle_entries.plate_number),
                    vehicle_image_path=COALESCE(excluded.vehicle_image_path, vehicle_entries.vehicle_image_path),
                    plate_image_path=COALESCE(excluded.plate_image_path, vehicle_entries.plate_image_path)""",
                entry,
            )

    def search(self, plate_query: str = "") -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM vehicle_entries WHERE COALESCE(plate_number, '') LIKE ? ORDER BY entry_time DESC",
                (f"%{plate_query.upper()}%",),
            ).fetchall()
        return [dict(row) for row in rows]

    def all_entries(self) -> list[dict]:
        return self.search("")