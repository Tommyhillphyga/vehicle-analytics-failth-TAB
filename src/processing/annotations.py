import cv2


def annotate_frame(frame, vehicles: list[dict], plates: list[dict]) -> object:
    output = frame.copy()
    for vehicle in vehicles:
        x1, y1, x2, y2 = (int(value) for value in vehicle["bbox"])
        label = f"{vehicle['class_name']} #{vehicle.get('track_id', '?')}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (35, 200, 80), 2)
        cv2.putText(output, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (35, 200, 80), 2)
    for plate in plates:
        x1, y1, x2, y2 = (int(value) for value in plate["bbox"])
        text = plate.get("text") or "plate"
        cv2.rectangle(output, (x1, y1), (x2, y2), (20, 180, 230), 2)
        cv2.line(output, (x1, max(20, y1-16)), (x2, max(20, y1-16)), (217, 156, 177), 35)
        cv2.putText(output, text, (x1+2, max(20, y1 - 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
    return output

