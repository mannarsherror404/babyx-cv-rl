# social/vision_affect.py
import cv2, math

try:
    import mediapipe as mp
except Exception:
    mp = None

MOUTH_L, MOUTH_R = 61, 291
LIP_TOP, LIP_BOTTOM = 13, 14

_face = None
if mp is not None and hasattr(mp.solutions, "face_mesh"):
    _mp_face = mp.solutions.face_mesh
    _face = _mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def _norm_dist(a, b, w, h):
    dx = (a.x - b.x) * w
    dy = (a.y - b.y) * h
    return math.hypot(dx, dy)

def estimate_affect(frame):
    """Return only neutral or happy, based on fixed MAR thresholds."""
    if _face is None:
        return "no_face", 0.0

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = _face.process(rgb)
    if not res.multi_face_landmarks:
        return "no_face", 0.0

    lmk = res.multi_face_landmarks[0].landmark

    mouth_w = _norm_dist(lmk[MOUTH_L], lmk[MOUTH_R], w, h)
    mouth_h = _norm_dist(lmk[LIP_TOP], lmk[LIP_BOTTOM], w, h)
    mar = mouth_h / (mouth_w + 1e-6)

    if mar < 0.010:
        return "neutral", mar
    elif 0.100 <= mar <= 0.900:
        return "happy", mar
    else:
        return "neutral", mar


def estimate_visual_affect(frame):
    """Return a numeric affect score for the main BabyX emotion loop."""
    label, score = estimate_affect(frame)
    if label == "happy":
        return min(1.0, max(0.0, float(score)))
    if label == "neutral":
        return 0.0
    return None

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Running vision affect (neutral vs happy)... press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        label, score = estimate_affect(frame)

        if label == "happy":
            color = (0, 255, 0)
        elif label == "neutral":
            color = (0, 255, 255)
        else:
            color = (200, 200, 200)

        cv2.putText(frame, f"{label.upper()} (MAR={score:.3f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Vision Affect", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
