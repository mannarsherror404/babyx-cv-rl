# social/audio_affect.py
import queue, threading, numpy as np

try:
    import sounddevice as sd
    import librosa
except Exception:
    sd = None
    librosa = None


class AudioAffect:
    def __init__(self, sr=16000, block_dur=0.25, hist_sec=0.75, debug=False):
        self.enabled = (sd is not None and librosa is not None)
        self.sr = sr
        self.block = int(sr * block_dur)
        self.hist_blocks = max(1, int(hist_sec / block_dur))
        self.q = queue.Queue()
        self._running = False
        self._score = 0.0
        self._buf = []
        self._lock = threading.Lock()
        self.stream = None
        self.thread = None
        self.debug = debug

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Stream status:", status)
        x = indata.copy()
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        self.q.put(x)

    def start(self, device=None):
        if not self.enabled or self._running:
            return
        try:
            self._running = True
            self.stream = sd.InputStream(
                channels=1, samplerate=self.sr, dtype="float32",
                blocksize=self.block, callback=self._callback, device=device
            )
            self.stream.start()
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
        except Exception as e:
            print("Mic error:", e)
            self.enabled = False
            self._running = False
            self.stream = None

    def stop(self):
        if not self._running:
            return
        self._running = False
        try:
            if self.stream:
                self.stream.stop(); self.stream.close()
        finally:
            self.stream = None

    def _worker(self):
        while self._running:
            try:
                x = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            self._buf.append(x)
            if len(self._buf) > self.hist_blocks:
                self._buf.pop(0)
            if not self._buf:
                continue
            y = np.concatenate(self._buf)
            if y.size < self.block:
                continue

            rms = float(np.sqrt(np.mean(y**2)))
            if librosa is None:
                score = 0.0
            else:
                zcr = float(np.mean(librosa.feature.zero_crossing_rate(y.reshape(1, -1))))
                sc  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sr)) / (self.sr/2))

                pos = 0.0
                if rms > 0.02 and sc > 0.35 and zcr > 0.08:
                    pos += (rms - 0.02) * 6.0 + (sc - 0.35) * 1.5

                neg = 0.0
                if rms > 0.025 and sc < 0.25:
                    neg += (rms - 0.025) * 5.0 + (0.25 - sc) * 1.2

                score = pos - neg
                score = float(np.clip(score, -1.0, 1.0))

                if self.debug:
                    print(f"[DEBUG] RMS={rms:.3f}, ZCR={zcr:.3f}, SC={sc:.3f}, Score={score:.2f}")

            with self._lock:
                self._score = 0.8 * self._score + 0.2 * score

    def get_score(self):
        if not self.enabled:
            return 0.0
        with self._lock:
            return float(self._score)


if __name__ == "__main__":
    import time
    affect = AudioAffect(debug=True)
    affect.start(device=2)

    try:
        print("Running audio affect (press CTRL+C to stop)...")
        while True:
            score = affect.get_score()
            print(f"Audio Affect Score: {score:.2f}")
            time.sleep(0.3)
    except KeyboardInterrupt:
        affect.stop()
        print("\nStopped.")
