# brain/emotion.py
import os, sys, time

# --- Fix path if running directly ---
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass


@dataclass
class EmotionConfig:
    POS_FEEDBACK   : float = +0.35
    NEG_FEEDBACK   : float = -0.45
    OBJECT_HAPPY   : dict  = None
    OBJECT_WARY    : dict  = None
    DECAY_PER_SEC  : float = 0.08
    CLAMP_MIN      : float = -1.0
    CLAMP_MAX      : float = +1.0

    def __post_init__(self):
        if self.OBJECT_HAPPY is None:
            self.OBJECT_HAPPY = {"duck": +0.12, "ball": +0.10}
        if self.OBJECT_WARY is None:
            self.OBJECT_WARY  = {"spider": -0.20, "knife": -0.30}


class EmotionState:
    def __init__(self):
        self.valence = 0.0
        self.last_t  = time.time()

    def _clamp(self, x, lo, hi):
        return max(lo, min(hi, x))

    def _decay(self, now, cfg: EmotionConfig):
        dt = max(0.0, now - self.last_t)
        if dt <= 0:
            return
        if self.valence > 0:
            self.valence = max(0.0, self.valence - cfg.DECAY_PER_SEC * dt)
        elif self.valence < 0:
            self.valence = min(0.0, self.valence + cfg.DECAY_PER_SEC * dt)
        self.last_t = now

    def update_from_events(self, looked_label: str, gestures: list[str], cfg: EmotionConfig):
        now = time.time()
        self._decay(now, cfg)

        lbl = (looked_label or "").lower()
        if lbl in cfg.OBJECT_HAPPY:
            self.valence += cfg.OBJECT_HAPPY[lbl]
        if lbl in cfg.OBJECT_WARY:
            self.valence += cfg.OBJECT_WARY[lbl]

        # Keep RL rewards (thumbs up/down) separate from baseline mood
        self.valence = self._clamp(self.valence, cfg.CLAMP_MIN, cfg.CLAMP_MAX)
        self.last_t = now

    def update_from_social(self, visual_affect: float | None, audio_affect: float | None, cfg: EmotionConfig):
        now = time.time()
        self._decay(now, cfg)

        w_vis, w_aud = 0.15, 0.20
        if isinstance(visual_affect, (int, float)):
            self.valence += w_vis * max(-1.0, min(1.0, float(visual_affect)))
        if isinstance(audio_affect, (int, float)):
            self.valence += w_aud * max(-1.0, min(1.0, float(audio_affect)))

        self.valence = self._clamp(self.valence, cfg.CLAMP_MIN, cfg.CLAMP_MAX)
        self.last_t = now

    def label(self):
        if self.valence >= 0.3:  return "happy"
        if self.valence <= -0.3: return "wary"
        return "neutral"

    def intensity01(self):
        return abs(self.valence)


def main():
    cfg = EmotionConfig()
    state = EmotionState()

    print("Running emotion demo... (CTRL+C to quit)")
    print("Type: duck / ball / spider / knife / neutral")
    while True:
        lbl = input("Enter object (or blank): ").strip().lower()
        if lbl == "q":
            break
        state.update_from_events(lbl, [], cfg)
        print(f"Valence: {state.valence:.2f} -> {state.label()} (intensity {state.intensity01():.2f})")


if __name__ == "__main__":
    main()
