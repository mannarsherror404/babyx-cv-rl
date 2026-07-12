# babyx/objects.py
import cv2
import torch
from ultralytics import YOLOWorld

# === Load YOLO-World-S model ===
# (Downloads automatically on first run ~250MB)
model = YOLOWorld("yolov8s-world.pt")

# Use Apple MPS GPU if available (M1/M2/M3 Macs), else CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

def get_objects(frame, conf=0.35, max_k=20):
    """
    Detect objects in a frame using YOLO-World-S.
    Returns list of dicts: {label, conf, bbox}.
    """
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=conf,
        max_det=max_k,
        device=device,
        verbose=False
    )

    out = []
    if results:
        r = results[0]
        for b in r.boxes:
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            label = r.names[cls]
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            out.append({
                "label": label,
                "conf": conf,
                "bbox": (x1, y1, x2 - x1, y2 - y1)
            })
    return out


# === Run standalone test ===
if __name__ == "__main__":
    print("Running YOLO-World-S object detection... (press 'q' to quit)")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        objects = get_objects(frame, conf=0.35, max_k=10)

        for d in objects:
            x, y, w, h = d["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{d['label']} {d['conf']:.2f}",
                        (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("BABY X - YOLO-World-S Objects", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
