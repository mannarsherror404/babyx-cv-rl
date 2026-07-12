# babyx/face.py
import cv2
import mediapipe as mp

_mp_face = mp.solutions.face_detection
_detector = _mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def get_faces(frame):
    """
    Input:  BGR frame
    Output: list of {'bbox': (x,y,w,h), 'conf': float}
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = _detector.process(rgb)
    out = []
    if res.detections:
        for d in res.detections:
            bb = d.location_data.relative_bounding_box
            x, y = int(bb.xmin * w), int(bb.ymin * h)
            ww, hh = int(bb.width * w), int(bb.height * h)
            out.append({"bbox": (x, y, ww, hh), "conf": float(d.score[0])})
    return out


# --- Standalone Test ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view
        faces = get_faces(frame)

        for i, f in enumerate(faces, start=1):
            x, y, w, h = f["bbox"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, f"Face {i} {f['conf']:.2f}",
                        (x, max(20, y-6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

        cv2.imshow("Face Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
