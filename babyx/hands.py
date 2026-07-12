# babyx/hands.py
import cv2, time, math
import mediapipe as mp

# --- MediaPipe Hands setup ---
_mp_hands = mp.solutions.hands
_hands = _mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,      # 0=faster, 1=more accurate
    max_num_hands=4,         # allow up to 4 hands in frame
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
_draw = mp.solutions.drawing_utils
_styles = mp.solutions.drawing_styles


# --- Gesture helpers ---
def _is_finger_folded(lm, tip, pip):
    """Check if finger is folded (tip below pip joint)."""
    return lm[tip].y > lm[pip].y


def _classify_thumb(lm, handedness, img_h):
    """
    Classify a single hand's thumb as up or down if other fingers are folded.
    Returns: "left_thumb_up", "left_thumb_down", "right_thumb_up", "right_thumb_down" or None
    """
    wrist_y = lm[0].y * img_h
    thumb_tip_y = lm[4].y * img_h

    # require other 4 fingers folded (like thumbs-up gesture)
    folded = all(_is_finger_folded(lm, tip, tip - 2)
                 for tip in [8, 12, 16, 20])

    if not folded:
        return None

    delta = img_h * 0.1
    if thumb_tip_y < wrist_y - delta:
        return f"{handedness.lower()}_thumb_up"
    if thumb_tip_y > wrist_y + delta:
        return f"{handedness.lower()}_thumb_down"
    return None


# --- State class ---
class GestureState:
    def __init__(self):
        pass


# --- Main API ---
def get_gestures(frame, state: GestureState, draw=True):
    """
    Detect only thumbs gestures.
    Returns one of:
    - left_thumb_up
    - left_thumb_down
    - right_thumb_up
    - right_thumb_down
    - both_up
    - both_down
    - left_up_right_down
    - right_up_left_down
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = _hands.process(rgb)

    gestures = []
    if res.multi_hand_landmarks and res.multi_handedness:
        per_hand = []
        for lm, hand_info in zip(res.multi_hand_landmarks, res.multi_handedness):
            handedness = hand_info.classification[0].label  # "Left"/"Right"
            if draw:
                _draw.draw_landmarks(
                    frame, lm, _mp_hands.HAND_CONNECTIONS,
                    _styles.get_default_hand_landmarks_style(),
                    _styles.get_default_hand_connections_style()
                )
            g = _classify_thumb(lm.landmark, handedness, h)
            if g:
                per_hand.append(g)

        # Resolve combinations
        if len(per_hand) == 1:
            gestures.extend(per_hand)
        elif len(per_hand) >= 2:
            if "left_thumb_up" in per_hand and "right_thumb_up" in per_hand:
                gestures.append("both_up")
            elif "left_thumb_down" in per_hand and "right_thumb_down" in per_hand:
                gestures.append("both_down")
            elif "left_thumb_up" in per_hand and "right_thumb_down" in per_hand:
                gestures.append("left_up_right_down")
            elif "right_thumb_up" in per_hand and "left_thumb_down" in per_hand:
                gestures.append("right_up_left_down")
            else:
                gestures.extend(per_hand)

    return gestures


# --- Standalone runner ---
if __name__ == "__main__":
    print("Running hand detection... (press 'q' to quit)")
    cap = cv2.VideoCapture(0)
    state = GestureState()

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)

        gestures = get_gestures(frame, state, draw=True)

        txt = ", ".join(gestures) if gestures else "-"
        cv2.putText(frame, f"{txt}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Hand Gestures", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
