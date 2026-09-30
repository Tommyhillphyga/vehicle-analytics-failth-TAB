from pathlib import Path

import cv2
import streamlit as st

from config.config import DEFAULT_SETTINGS, Settings
from src.database.database import VehicleDatabase
from src.detection.plate_detector import PlateDetector
from src.detection.vehicle_detector import VehicleDetector
from src.ocr.plate_ocr import PlateOCR
from src.processing.video_processor import VideoProcessor


st.set_page_config(page_title="Vehicle Analytics", page_icon="VA", layout="centered")
settings = DEFAULT_SETTINGS
settings.ensure_directories()
database = VehicleDatabase(settings.database_path)


@st.cache_resource(show_spinner=False)
def load_models(config: Settings):
    return (
        VehicleDetector(config.vehicle_model_path, config.vehicle_confidence, config.iou_threshold),
        PlateDetector(config.plate_model_path, config.plate_confidence),
        PlateOCR(config.ocr_confidence),
    )


def show_sidebar(entries: list[dict]) -> None:
    with st.sidebar:
        st.header("Detected vehicles")
        if not entries:
            st.caption("Processed vehicles will appear here.")
        for entry in entries:
            st.divider()
            st.caption(f"Vehicle #{entry['track_id']} · {entry['vehicle_type'].title()}")
            st.write(f"Plate: **{entry.get('plate_number') or 'Processing...'}**")


st.title("Vehicle Analytics")
st.caption("Upload a recorded road video to detect, track, read, and search vehicles.")
uploaded = st.file_uploader("Upload vehicle video", type=["mp4", "avi", "mov", "mkv"])
process = st.button("Process video", type="primary", disabled=uploaded is None)
st.subheader("Live processing")
frame_slot = st.empty()
previous_result = st.session_state.get("last_result")
previous_counts = previous_result.get("counts", {}) if previous_result else {}
metrics = st.columns(4)
total_metric = metrics[0].empty()
car_metric = metrics[1].empty()
bus_metric = metrics[2].empty()
truck_metric = metrics[3].empty()
total_metric.metric("Total vehicles", len(previous_result["tracks"]) if previous_result else 0)
car_metric.metric("Cars", previous_counts.get("car", 0))
bus_metric.metric("Buses", previous_counts.get("bus", 0))
truck_metric.metric("Trucks", previous_counts.get("truck", 0))

if process and uploaded:
    video_path = Path(settings.video_dir) / Path(uploaded.name).name
    video_path.write_bytes(uploaded.getbuffer())
    try:
        with st.spinner("Loading detection and OCR models..."):
            vehicle_detector, plate_detector, ocr = load_models(settings)
        processor = VideoProcessor(settings, database, vehicle_detector, plate_detector, ocr)
        progress = st.progress(0, text="Processing frames")

        def render_frame(frame, state):
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="content")
            counts = state["counts"]
            total_metric.metric("Total vehicles", state["total_vehicles"])
            car_metric.metric("Cars", counts.get("car", 0))
            bus_metric.metric("Buses", counts.get("bus", 0))
            truck_metric.metric("Trucks", counts.get("truck", 0))
            total = state["total_frames"]
            if total:
                progress.progress(min(1.0, (state["frame"] + 1) / total), text=f"Processing frame {state['frame'] + 1} of {total}")

        result = processor.process(str(video_path), render_frame)
        st.session_state["last_result"] = result
        progress.progress(1.0, text="Processing complete")
        st.success(f"Processed {result['frames']} frames and found {len(result['tracks'])} unique vehicles.")
    except Exception as error:
        st.error(f"Processing failed: {error}")
    finally:
        frame_slot.empty()

st.subheader("Search by plate")
query = st.text_input("License plate contains", placeholder="ABC123")
matches = database.search(query)
if query and not matches:
    st.info(f"No vehicle found for plate {query.upper()}")
for match in matches:
    with st.container(border=True):
        st.write(f"**{match.get('plate_number') or 'Unknown plate'}** · {match['vehicle_type'].title()}")
        st.caption(f"Track ID {match['track_id']} · Entry {match['entry_time']}")
        if match.get("vehicle_image_path"):
            st.image(match["vehicle_image_path"], width=180)