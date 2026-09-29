# Vehicle Analytics Dashboard

An MVP Streamlit application for processing uploaded road videos with YOLOv8 vehicle detection, ByteTrack IDs, a configurable license-plate detector, EasyOCR, live annotated-frame preview, and SQLite-backed plate search.

## Features

- Detects COCO `car`, `bus`, and `truck` classes with configurable YOLO confidence and IoU thresholds.
- Uses Ultralytics ByteTrack persistence for unique vehicle IDs.
- Detects plates with a separate YOLO model, associates plate boxes to vehicles by containment, and reads them with EasyOCR.
- Keeps the highest-confidence/frequently observed plate reading per track.
- Saves one representative vehicle crop and the best available plate crop.
- Stores every unique track in SQLite and supports partial plate search.
- Shows annotated frames during processing, vehicle-type counts, and detected vehicle entries in the sidebar. The live frame preview clears when processing ends.
- Handles unreadable or missing plates without dropping the vehicle record.

## Architecture

`app.py` is the UI only. The processing path is separated into `src/detection`, `src/tracking`, `src/ocr`, `src/processing`, `src/database`, and `src/utils`. Heavy packages are imported by their adapters so utility and database tests can run without model initialization.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Models

Create `models/` and place the supplied plate detector at:

```text
models/license_plate_detector.pt
```

The default vehicle path is `models/yolov8n.pt`. Ultralytics downloads `yolov8n.pt` automatically on first use when it is not present. Change both paths in `config/config.py` or replace `DEFAULT_SETTINGS` with environment-backed settings for deployment.

The plate model should output plate bounding boxes. The default implementation expects Ultralytics YOLO results.

## Run

```bash
streamlit run app.py
```

Upload an MP4, AVI, MOV, or MKV file and select **Process video**. The dashboard updates the annotated frame preview and processing progress as frames are processed; it does not display a post-processing frame gallery. The SQLite database and image directories are created automatically under `data/`.

## Configuration

Edit `Settings` in `config/config.py`:

```python
vehicle_confidence = 0.40
plate_confidence = 0.40
ocr_confidence = 0.50
process_every_n_frames = 1
```

For each processed frame, plate detection runs on each detected vehicle crop, then OCR runs on the detected plate crops. `vehicle_analytics.db` contains the `vehicle_entries` table with track ID, vehicle type, plate, entry time, and crop paths.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

The tests cover plate normalization and safe filenames, unique track counting, SQLite upsert/search, and missing-plate behavior. Model-backed tests require the detector/OCR dependencies and actual model files.

## Troubleshooting and limitations

- If model loading fails, verify `ultralytics` is installed and the configured model paths are readable.
- If EasyOCR fails to initialize, verify that PyTorch is installed for the active Python/platform combination and allow EasyOCR to download its recognition models on first use.
- A missing or unreadable plate still creates a vehicle entry with a null plate.
- Entry time is the video-relative first-seen timestamp formatted as `HH:MM:SS`; it is not wall-clock capture time.
- This MVP processes uploaded files sequentially and does not implement authentication, RTSP streaming, speed, make/model, color, face, or driver recognition.