# brain/attention.py
import os, sys, time, math, cv2

# --- Fix path if running directly ---
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from babyx.face import get_faces
from babyx.objects import get_objects


class AttentionConfig:
    FACE_BONUS       = 1.5
    CONF_WEIGHT      = 1.0
    AREA_WEIGHT      = 0.0006
    CENTER_WEIGHT    = 0.8
    NOVELTY_BOOST    = 0.15
    NOVELTY_COOLDOWN = 3.0

    DWELL_MIN_SEC    = 0.6
    DWELL_MAX_SEC    = 2.0
    EPSILON          = 0.08

    SMOOTH_ALPHA     = 0.30


class AttentionState:
    def __init__(self):
        self.current_label = None
        self.current_kind  = None
        self.current_xy    = None
        self.current_score = 0.0
        self.last_switch_t = 0.0
        self.last_seen     = {}

    def _ema(self, old_xy, new_xy, alpha):
        if old_xy is None:
            return new_xy
        ox, oy = old_xy
        nx, ny = new_xy
        return ((1 - alpha) * ox + alpha * nx,
                (1 - alpha) * oy + alpha * ny)


def _center_bias(x, y, W, H):
    cx, cy = W / 2.0, H / 2.0
    dx = (x - cx) / (W / 2.0)
    dy = (y - cy) / (H / 2.0)
    dist = math.hypot(dx, dy)
    return max(0.0, 1.0 - dist)


def _area_score(w, h, weight):
    return weight * (w * h)


def _base_candidate(label, kind, xy, w, h, conf, W, H, cfg, now, last_seen):
    cx, cy = xy
    score = 0.0
    if conf is not None:
        score += cfg.CONF_WEIGHT * float(conf)
    score += _area_score(w, h, cfg.AREA_WEIGHT)
    score += cfg.CENTER_WEIGHT * _center_bias(cx, cy, W, H)
    if kind == "face":
        score *= cfg.FACE_BONUS
    last = last_seen.get(label, 0.0)
    if now - last > cfg.NOVELTY_COOLDOWN:
        score *= (1.0 + cfg.NOVELTY_BOOST)
    return score


def _enumerate_candidates(faces, objects, frame_shape, cfg, now, last_seen):
    H, W = frame_shape[:2]
    cands = []

    for i, f in enumerate(faces, start=1):
        x, y, w, h = f["bbox"]
        cx, cy = x + w / 2.0, y + h / 2.0
        label = f"face_{i}"
        conf  = f.get("conf", 0.8)
        score = _base_candidate(label, "face", (cx, cy), w, h, conf, W, H, cfg, now, last_seen)
        cands.append({"label": label, "kind": "face", "xy": (cx, cy), "w": w, "h": h,
                      "conf": float(conf), "score": float(score)})

    for o in objects:
        x, y, w, h = o["bbox"]
        cx, cy = x + w / 2.0, y + h / 2.0
        label = str(o.get("label", "object"))
        conf  = o.get("conf", 0.5)
        score = _base_candidate(label, "object", (cx, cy), w, h, conf, W, H, cfg, now, last_seen)
        cands.append({"label": label, "kind": "object", "xy": (cx, cy), "w": w, "h": h,
                      "conf": float(conf), "score": float(score)})

    cx, cy = W / 2.0, H / 2.0
    idle_score = 0.45 + 0.55 * _center_bias(cx, cy, W, H)
    cands.append({"label": "idle", "kind": "idle", "xy": (cx, cy), "w": 0, "h": 0,
                  "conf": 1.0, "score": float(idle_score)})

    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def update_attention(faces, objects, frame_shape, state, cfg=None):
    if cfg is None:
        cfg = AttentionConfig()
    now = time.time()

    cands = _enumerate_candidates(faces, objects, frame_shape, cfg, now, state.last_seen)
    if not cands:
        H, W = frame_shape[:2]
        state.current_xy = (W / 2.0, H / 2.0)
        state.current_label = "idle"
        state.current_kind  = "idle"
        state.current_score = 0.5
        state.last_switch_t = now
        return {"xy": state.current_xy, "label": state.current_label, "kind": state.current_kind,
                "score": state.current_score, "dwell_elapsed": 0.0,
                "dwell_min": cfg.DWELL_MIN_SEC, "dwell_max": cfg.DWELL_MAX_SEC,
                "sticking": True, "best_label": "idle", "best_score": state.current_score}

    for c in cands:
        state.last_seen.setdefault(c["label"], 0.0)

    best = cands[0]
    dwell_elapsed = now - state.last_switch_t if state.last_switch_t else 0.0

    sticking = False
    if state.current_label is not None:
        if dwell_elapsed < cfg.DWELL_MIN_SEC:
            sticking = True
        elif dwell_elapsed < cfg.DWELL_MAX_SEC and state.current_label == best["label"]:
            sticking = True

    if sticking:
        still_there = any(c["label"] == state.current_label for c in cands)
        target = next((c for c in cands if c["label"] == state.current_label), best) if still_there else best
    else:
        target = best
        if len(cands) > 1 and (state.current_label != cands[1]["label"]) and (math.fmod(now, 1.0) < cfg.EPSILON):
            target = cands[1]
        if state.current_label != target["label"]:
            state.last_switch_t = now
            dwell_elapsed = 0.0

    state.last_seen[target["label"]] = now
    smoothed_xy = state._ema(state.current_xy, target["xy"], cfg.SMOOTH_ALPHA)
    state.current_xy    = smoothed_xy
    state.current_label = target["label"]
    state.current_kind  = target["kind"]
    state.current_score = target["score"]

    return {"xy": smoothed_xy, "label": state.current_label, "kind": state.current_kind,
            "score": float(state.current_score), "dwell_elapsed": float(dwell_elapsed),
            "dwell_min": float(cfg.DWELL_MIN_SEC), "dwell_max": float(cfg.DWELL_MAX_SEC),
            "sticking": bool(sticking), "best_label": best["label"], "best_score": float(best["score"])}


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FATAL] Camera not available")
        return

    state = AttentionState()
    cfg   = AttentionConfig()

    print("Running attention demo... (press 'q' to quit)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        faces = get_faces(frame)
        objs  = get_objects(frame, conf=0.25, max_k=10)

        att = update_attention(faces, objs, frame.shape, state, cfg)
        tx, ty = map(int, att["xy"])

        cv2.drawMarker(frame, (tx, ty), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
        cv2.circle(frame, (tx, ty), 6, (255, 255, 255), 2)
        cv2.putText(frame, f"Look: {att['label']} ({att['kind']}) s={att['score']:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Attention Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
