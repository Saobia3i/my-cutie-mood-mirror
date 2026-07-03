# Mood Mirror 🐱✨

A cute webcam toy: your facial expression is mirrored live onto a cartoon
cat character in the center of the screen, with soft particle effects and
gentle encouraging messages.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Press **q** or **Esc** to quit.

## How it works

- `face_tracker.py` — MediaPipe Face Mesh extracts 468 facial landmarks per frame.
- `expression_classifier.py` — rule-based logic (no ML training needed) turns
  landmark distances into one of 4 moods: `neutral`, `happy`, `surprised`, `sleepy`.
  Uses eye-aspect-ratio for blink/sleepy detection, mouth-open ratio for surprise,
  and mouth-width ratio for smile detection. Smoothed over a short frame window
  so the mood doesn't flicker.
- `character.py` — loads the 4 sprite states and renders the character centered
  on screen with a soft crossfade between moods and a gentle idle bounce.
- `particles.py` — hearts (happy), stars (surprised), floating "z" (sleepy).
- `main.py` — glues it all together: dims/blurs the raw webcam feed as backdrop,
  tints it per mood, and overlays the character + particles + message popups.

## Tuning

If mood detection feels off for your lighting/face, adjust the thresholds in
`expression_classifier.py`:

```python
self.EAR_SLEEPY_THRESH = 0.16
self.MOUTH_OPEN_SURPRISE = 0.55
self.SMILE_RATIO_THRESH = 0.42
```

Lower `EAR_SLEEPY_THRESH` if it triggers sleepy too easily; raise
`SMILE_RATIO_THRESH` if it's too eager to call things a smile, etc.

## Assets

`assets/neutral.png`, `happy.png`, `surprised.png`, `sleepy.png` — cropped from
the provided cat sprite sheet. Swap these with your own PNGs (transparent
background) to reskin the character; just keep the same filenames.
