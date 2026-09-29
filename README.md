# Vehicle Analytics Dashboard

An MVP Streamlit application for processing uploaded road videos with YOLOv8 vehicle detection, ByteTrack IDs, a configurable license-plate detector, EasyOCR, live annotated-frame preview, and SQLite-backed plate search.


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

