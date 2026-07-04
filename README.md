---
title: Mood Mirror
emoji: 🐱
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
---

# Mood Mirror

A cute webcam toy: your facial expression is mirrored live onto a cartoon cat
character with soft particle effects and gentle encouraging messages.

Allow camera access, then smile, look surprised, squint sleepily, furrow your
brow, or pout to watch the cat react.

## Deploy

This project is ready for Hugging Face Spaces.

1. Push this repo to GitHub.
2. Create a new Hugging Face Space.
3. Choose `Gradio` as the SDK.
4. Connect/import the GitHub repo, or upload these files directly.

Hugging Face will use:

- `app.py` as the app entry point.
- `requirements.txt` for dependencies.
- `face_landmarker.task` and `assets/` as bundled runtime assets.

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

For the native OpenCV desktop version:

```bash
python main.py
```

Press `q` or `Esc` to quit the desktop version.

## How It Works

- `face_tracker.py` uses MediaPipe Face Landmarker.
- `expression_classifier.py` uses rule-based expression detection.
- `character.py` renders the mood sprite.
- `particles.py` draws mood-specific particle effects.
- `app.py` wraps everything in a browser-based Gradio app for deployment.
