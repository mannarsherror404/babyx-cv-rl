# main_babyx.py
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
from babyx.face import get_faces
from babyx.hands import get_gestures, GestureState
from babyx.objects import get_objects
from brain.attention import AttentionState, AttentionConfig, update_attention
from brain.emotion import EmotionState, EmotionConfig
from social.vision_affect import estimate_visual_affect
from social.audio_affect import AudioAffect

WINDOW_NAME = "BABY X - modes (1/2/3/4 or drag slider)"

BTN_Y = 60
BTN_W = 120
BTN_H = 40
BTN_GAP = 12
BTN_TEXT_Y = BTN_Y + 28

def button_rect(i):
    x1 = 10 + (i - 1) * (BTN_W + BTN_GAP)
    return (x1, BTN_Y, x1 + BTN_W, BTN_Y + BTN_H)

def draw_button(frame, i, label, active=False):
    x1,y1,x2,y2 = button_rect(i)
    color = (0,255,0) if active else (120,120,120)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, f"{i}: {label}", (x1 + 10, BTN_TEXT_Y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def inside(px, py, rect):
    x1,y1,x2,y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2

def open_cam():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FATAL] Camera not available. Check macOS camera permissions for Terminal/VSCode.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def color_for_label(label):
    h = abs(hash(label)) % 255
    return (int((h*97) % 255), int((h*57) % 255), int((h*23) % 255))

def draw_dwell_hud(frame, att):
    status  = "sticking" if att.get("sticking") else "free"
    elapsed = float(att.get("dwell_elapsed", 0.0))
    dmin    = float(att.get("dwell_min", 0.0))
    dmax    = float(att.get("dwell_max", 1.0))

    txt = f"Dwell: {elapsed:.1f}s  (min {dmin:.1f}s / max {dmax:.1f}s) [{status}]"
    cv2.putText(frame, txt, (10, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)

    x0, y0, w, h = 10, 118, 300, 10
    cv2.rectangle(frame, (x0, y0), (x0+w, y0+h), (120,120,120), 1)
    dmax = max(1e-6, dmax)
    p = max(0.0, min(1.0, elapsed / dmax))
    cv2.rectangle(frame, (x0, y0), (x0+int(w*p), y0+h), (80,220,120), -1)
    m = max(0.0, min(1.0, dmin / dmax))
    x_min = x0 + int(w*m)
    cv2.line(frame, (x_min, y0-3), (x_min, y0+h+3), (0,255,255), 2)

def main():
    cap = open_cam()
    if cap is None:
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 900, 700)

    mode = 1  # 1=Face, 2=Hands, 3=Objects, 4=All
    cv2.createTrackbar("Mode 1=Face 2=Hands 3=Obj 4=All", WINDOW_NAME, mode, 4, lambda v: None)

    gstate     = GestureState()
    att_state  = AttentionState()
    att_cfg    = AttentionConfig()
    emo_state  = EmotionState()
    emo_cfg    = EmotionConfig()

    audio_aff = AudioAffect(sr=16000)
    audio_aff.start()

    def on_mouse(event, x, y, flags, userdata):
        nonlocal mode
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in (1,2,3,4):
                if inside(x, y, button_rect(i)):
                    mode = i
                    cv2.setTrackbarPos("Mode 1=Face 2=Hands 3=Obj 4=All", WINDOW_NAME, mode)
                    break
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("Controls: 1=Face  2=Hands  3=Objects  4=All  q=Quit")
    print("Or drag the slider / click the on-screen buttons.")
    print("Make sure the window title is:", WINDOW_NAME)

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)

            tb = cv2.getTrackbarPos("Mode 1=Face 2=Hands 3=Obj 4=All", WINDOW_NAME)
            if tb in (1,2,3,4):
                mode = tb

            faces, gestures, objs = [], [], []

            if mode == 1:
                faces = get_faces(frame)
            elif mode == 2:
                gestures = get_gestures(frame, gstate)
            elif mode == 3:
                objs = get_objects(frame, conf=0.25, max_k=30)
            elif mode == 4:
                faces = get_faces(frame)
                gestures = get_gestures(frame, gstate)
                objs = get_objects(frame, conf=0.25, max_k=30)

            for i,f in enumerate(faces,1):
                x,y,w,h = f["bbox"]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                cv2.putText(frame,f"Face {i} {f['conf']:.2f}",(x,max(20,y-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)

            for d in objs:
                x,y,w,h = d["bbox"]
                color = color_for_label(d["label"])
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,f"{d['label']} {d['conf']:.2f}",
                            (x, max(20,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            att = update_attention(faces, objs, frame.shape, att_state, att_cfg)
            tx, ty = map(int, att["xy"])

            visual_aff = estimate_visual_affect(frame) if faces else None
            audio_score = audio_aff.get_score()
            emo_state.update_from_social(visual_aff, audio_score, emo_cfg)

            emo_label = emo_state.label()
            cross_col = (255,255,255)
            if   emo_label == "happy": cross_col = (180,255,180)
            elif emo_label == "wary":  cross_col = (80,180,255)

            cv2.drawMarker(frame, (tx, ty), cross_col,
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
            cv2.circle(frame, (tx, ty), 6, cross_col, 2)
            cv2.putText(frame, f"Look: {att['label']} ({att['kind']}) s={att['score']:.2f}",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            draw_dwell_hud(frame, att)

            vis_str = "-" if visual_aff is None else f"{visual_aff:+.2f}"
            emo_txt = f"Emotion: {emo_label}  v={emo_state.valence:+.2f}  vis={vis_str}  aud={audio_score:+.2f}"
            cv2.putText(frame, emo_txt, (10, 144),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,255), 2)

            cv2.putText(frame, f"Mode: {mode}  [q=Quit | 1=Face 2=Hands 3=Objects 4=All]",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            g_text = ", ".join(gestures) if gestures else "-"
            cv2.putText(frame, f"Gestures: {g_text}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,220,255), 2)

            draw_button(frame, 1, "Face",   active=(mode==1))
            draw_button(frame, 2, "Hands",  active=(mode==2))
            draw_button(frame, 3, "Objects",active=(mode==3))
            draw_button(frame, 4, "All",    active=(mode==4))

            cv2.imshow(WINDOW_NAME, frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q')): break
            elif k == ord('1'):
                mode = 1; cv2.setTrackbarPos("Mode 1=Face 2=Hands 3=Obj 4=All", WINDOW_NAME, mode)
            elif k == ord('2'):
                mode = 2; cv2.setTrackbarPos("Mode 1=Face 2=Hands 3=Obj 4=All", WINDOW_NAME, mode)
            elif k == ord('3'):
                mode = 3; cv2.setTrackbarPos("Mode 1=Face 2=Hands 3=Obj 4=All", WINDOW_NAME, mode)
            elif k == ord('4'):
                mode = 4; cv2.setTrackbarPos("Mode 1=Face 2=Hands 3=Obj 4=All", WINDOW_NAME, mode)

    finally:
        audio_aff.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
