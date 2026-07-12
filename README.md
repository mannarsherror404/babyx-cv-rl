# BabyX CV + RL Perception Prototype

BabyX is a real-time computer vision and reinforcement-feedback prototype that combines camera-based face, hand, object, attention, and affect signals into a single interactive perception loop.

## What It Does

- Detects faces with MediaPipe.
- Detects hand gestures, including thumbs-up and thumbs-down feedback.
- Detects objects using YOLO-World through Ultralytics.
- Maintains a lightweight attention state for what BabyX is looking at.
- Estimates simple visual/audio affect signals for an emotion loop.
- Shows a real-time OpenCV interface with selectable modes.

## Project Structure

- `main_babyx.py` - main real-time demo loop.
- `babyx/` - face, hand, and object perception modules.
- `brain/` - attention and emotion state logic.
- `social/` - visual and audio affect modules.
- `Eyes/` - experimental combined perception module.
- `BabyX System Flow.png` - system flow diagram.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main_babyx.py
```

Controls:

- `1` - Face mode
- `2` - Hands mode
- `3` - Objects mode
- `4` - All mode
- `q` - Quit

## Notes

Model weights such as YOLO `.pt` files are intentionally not committed. Ultralytics can download the required model on first use, or you can place the model file locally beside the script.
