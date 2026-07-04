"""
app.py
Hugging Face Spaces entry point -- Gradio version of Mood Mirror.

Unlike main.py (which opens a native cv2 window and reads the webcam directly),
this version receives webcam frames from the user's BROWSER via Gradio's
streaming Image component, processes each frame server-side using the same
face_tracker / expression_classifier / character / particles pipeline, and
streams the processed frame back to display in the browser.

Each browser session gets its own state (own FaceTracker, Character, etc.)
via gr.State, so multiple people can use the Space at once without
interfering with each other.
"""

import random
import cv2
import numpy as np
import gradio as gr

from face_tracker import FaceTracker
from expression_classifier import ExpressionClassifier
from character import Character
from particles import ParticleSystem
from main import (
    ASSETS_DIR,
    MOOD_MESSAGES,
    apply_mood_tint,
    dim_and_blur_background,
    draw_message,
)


def _new_session():
    return {
        "tracker": FaceTracker(),
        "classifier": ExpressionClassifier(),
        "character": Character(ASSETS_DIR, base_size=340),
        "particles": ParticleSystem(),
        "current_mood": "neutral",
        "current_message": "",
        "message_timer": 0,
        "frame_count": 0,
        "tongue_timer": 0,
    }


def trigger_tongue(session):
    if session is None:
        session = _new_session()
    session["tongue_timer"] = 45
    return session


def process_frame(frame, session):
    if frame is None:
        return None, session

    if session is None:
        session = _new_session()

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    target_w = 480
    fh, fw = frame_bgr.shape[:2]
    if fw > target_w:
        scale = target_w / fw
        frame_bgr = cv2.resize(frame_bgr, (target_w, int(fh * scale)), interpolation=cv2.INTER_AREA)

    frame_bgr = cv2.flip(frame_bgr, 1)
    h, w = frame_bgr.shape[:2]

    landmarks = session["tracker"].process(frame_bgr)
    mood = session["classifier"].update(landmarks)

    if session["tongue_timer"] > 0:
        mood = "tongue"
        session["tongue_timer"] -= 1

    session["character"].set_mood(mood)

    bg = dim_and_blur_background(frame_bgr)
    bg = apply_mood_tint(bg, mood)

    cx, cy = w // 2, h // 2
    bg, (char_x, char_y) = session["character"].render(bg, cx, cy)

    if mood != session["current_mood"]:
        session["particles"].trigger(mood, char_x, char_y)
        session["current_message"] = random.choice(MOOD_MESSAGES.get(mood, [""]))
        session["message_timer"] = 60
        session["current_mood"] = mood

    session["particles"].maybe_ambient(mood, char_x, char_y, session["frame_count"])
    bg = session["particles"].update_and_draw(bg)

    if session["message_timer"] > 0:
        draw_message(bg, session["current_message"], cx, char_y + 220)
        session["message_timer"] -= 1

    session["frame_count"] += 1

    out_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    return out_rgb, session


# ---------------------------------------------------------------------------
# IMPORTANT: the outer <body>/<html> of a Gradio app lives OUTSIDE the
# <gradio-app> web component's Shadow DOM. CSS passed via css=... is injected
# INSIDE that shadow root, so it can NEVER reach html/body -- that's why the
# page background outside the centered container was staying black/dark
# (Gradio falls back to its dark-mode default, #0b0f19, when the visitor's
# OS/browser is in dark mode). The only way to control that outer background
# is through the THEME's body_background_fill / body_background_fill_dark,
# which Gradio renders into a <style> tag in the real page <head>.
# ---------------------------------------------------------------------------
CREAM_GRADIENT = "linear-gradient(160deg, #fff9ec 0%, #ffedc2 45%, #ffd9a0 100%)"

THEME = gr.themes.Soft(primary_hue="amber", secondary_hue="orange", neutral_hue="stone").set(
    body_background_fill=CREAM_GRADIENT,
    body_background_fill_dark=CREAM_GRADIENT,  # <-- fixes the black bars/gaps
)

CUSTOM_CSS = """
* { font-family: 'Trebuchet MS', 'Segoe UI', sans-serif !important; }

/* .gradio-container itself is inside the shadow root, so this part still
   works fine as before -- keep it in sync with the theme gradient above. */
.gradio-container {
    background: linear-gradient(160deg, #fff9ec 0%, #ffedc2 45%, #ffd9a0 100%) !important;
    max-width: 900px !important;
    width: 100% !important;
    min-height: 100vh !important;
    margin: 0 auto !important;
    box-sizing: border-box !important;
}

#title-block {
    text-align: center;
    padding: 8px 4px 4px 4px;
}
#title-block h1 {
    font-size: clamp(1.4em, 5vw, 2.1em) !important;
    background: linear-gradient(90deg, #d2691e, #ff8c00, #f4b400);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px !important;
}
#title-block p {
    color: #8a6a3c !important;
    font-size: clamp(0.8em, 2.5vw, 0.95em) !important;
}

.cam-card, .out-card {
    border-radius: 22px !important;
    overflow: hidden;
    box-shadow: 0 8px 22px rgba(210, 140, 40, 0.22) !important;
    border: 3px solid #fff3d6 !important;
    background: #fffaf0 !important;
}

/* Force the webcam/output boxes to scale with a fixed aspect ratio instead
   of Gradio's default fixed pixel height, so they shrink/grow smoothly on
   any screen instead of jumping to a fixed size. */
.cam-card .image-container,
.out-card .image-container,
.cam-card video,
.out-card img,
.cam-card canvas,
.out-card canvas {
    aspect-ratio: 4 / 3 !important;
    width: 100% !important;
    height: auto !important;
    object-fit: cover !important;
}

.responsive-row {
    flex-wrap: wrap !important;
}

#tongue-btn {
    border-radius: 999px !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #ffb347, #ff8c42) !important;
    color: #5c3a1e !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(255, 140, 66, 0.45) !important;
    transition: transform 0.15s ease !important;
    width: 100%;
    max-width: 320px;
}
#tongue-btn:hover {
    transform: scale(1.04);
}

/* Desktop (>900px): camera + output side by side.
   Tablet & mobile (<=900px): stack vertically, full width -- easier to use
   on touch screens and smaller viewports. */
@media (max-width: 900px) {
    .responsive-row {
        flex-direction: column !important;
    }
    .cam-card, .out-card {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }
    .gradio-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
}

/* Mobile phones specifically: tighter spacing, full-width tap-friendly button */
@media (max-width: 480px) {
    #title-block h1 { font-size: 1.3em !important; }
    #tongue-btn {
        max-width: 100% !important;
        font-size: 0.95em !important;
        padding: 10px 0 !important;
    }
    .cam-card .image-container,
    .out-card .image-container {
        aspect-ratio: 3 / 4 !important;
    }
}

@media (max-width: 360px) {
    .gradio-container {
        padding-left: 6px !important;
        padding-right: 6px !important;
    }
}
"""

with gr.Blocks(title="Mood Mirror 🐱", theme=THEME, css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="title-block"):
        gr.Markdown(
            "# 🐱 Mood Mirror\n"
            "Your face, mirrored onto a cute tabby cat, in real time~ 🐾"
        )

    with gr.Row(elem_classes="responsive-row", equal_height=True):
        with gr.Column(elem_classes="cam-card", min_width=260, scale=1):
            cam = gr.Image(sources=["webcam"], streaming=True, label="📷 You", type="numpy")
        with gr.Column(elem_classes="out-card", min_width=260, scale=1):
            out = gr.Image(label="🐾 Mood Mirror", type="numpy")

    with gr.Row():
        tongue_btn = gr.Button("😜 Stick out tongue!", elem_id="tongue-btn", size="lg")

    gr.Markdown(
        "<div style='text-align:center; color:#a9895c; font-size:0.85em; margin-top:6px;'>"
        "made with 🧡 using MediaPipe + OpenCV + Gradio</div>"
    )

    session_state = gr.State(None)
    tongue_btn.click(fn=trigger_tongue, inputs=[session_state], outputs=[session_state])

    cam.stream(
        fn=process_frame,
        inputs=[cam, session_state],
        outputs=[out, session_state],
        time_limit=300,
        stream_every=0.15,
        concurrency_limit=12,
    )

if __name__ == "__main__":
    demo.launch()
