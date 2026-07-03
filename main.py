"""
main.py
Mood Mirror -- a cute webcam toy that mirrors your facial expression
onto a little cartoon cat character in the center of the screen.

Controls:
  q       -> quit
  ESC     -> quit
"""

import os
import random
import time
import cv2
import numpy as np

from face_tracker import FaceTracker
from expression_classifier import ExpressionClassifier
from character import Character
from particles import ParticleSystem


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Soft background tint per mood (BGR), applied as a gentle wash, not a hard overlay
MOOD_TINT = {
    "neutral": (60, 40, 20),
    "happy": (30, 70, 60),
    "surprised": (60, 60, 10),
    "sleepy": (50, 30, 30),
    "angry": (20, 20, 70),
    "sad": (55, 25, 15),
}

MOOD_MESSAGES = {
    "happy": ["You're glowing today! ✨", "That smile though 😸", "Purr-fect mood!"],
    "surprised": ["Oh! Didn't see that coming~", "Whoa there!", "Big reaction energy!"],
    "sleepy": ["Someone's cozy...", "Nap time? 😽", "Soft and sleepy~"],
    "angry": ["Ooh, feisty! 😾", "Deep breaths~", "Hiss detected!"],
    "sad": ["Sending soft hugs 💙", "It's okay to feel this.", "Here with you."],
    "neutral": ["Just vibing.", "Calm and steady.", ""],
}


def apply_mood_tint(frame, mood, strength=0.15):
    tint = np.full(frame.shape, MOOD_TINT.get(mood, (0, 0, 0)), dtype=np.uint8)
    return cv2.addWeighted(frame, 1 - strength, tint, strength, 0)


def dim_and_blur_background(frame, dim=0.55, blur_ksize=21):
    blurred = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)
    dark = (blurred.astype(np.float32) * dim).astype(np.uint8)
    return dark


def draw_message(frame, text, center_x, y, alpha=1.0):
    if not text:
        return frame
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = center_x - tw // 2

    overlay = frame.copy()
    pad = 14
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                  (245, 245, 245), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55 * alpha, frame, 1 - 0.55 * alpha, 0, dst=frame)
    cv2.putText(frame, text, (x, y), font, scale, (60, 50, 90), thickness, cv2.LINE_AA)
    return frame


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend -- fixes freeze/pause issues on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep buffer small so frames don't lag behind

    tracker = FaceTracker()
    classifier = ExpressionClassifier()
    character = Character(ASSETS_DIR, base_size=340)
    particles = ParticleSystem()

    current_mood = "neutral"
    current_message = ""
    message_timer = 0
    frame_count = 0

    window_name = "Mood Mirror"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        landmarks = tracker.process(frame)
        mood = classifier.update(landmarks)
        character.set_mood(mood)
        bg = dim_and_blur_background(frame)
        bg = apply_mood_tint(bg, mood)

        cx, cy = w // 2, h // 2
        bg, (char_x, char_y) = character.render(bg, cx, cy)

        if mood != current_mood:
            particles.trigger(mood, char_x, char_y)
            current_message = random.choice(MOOD_MESSAGES.get(mood, [""]))
            message_timer = 60

            current_mood = mood

        particles.maybe_ambient(mood, char_x, char_y, frame_count)
        bg = particles.update_and_draw(bg)

        if message_timer > 0:
            draw_message(bg, current_message, cx, char_y + 220)
            message_timer -= 1
            # --- DEBUG OVERLAY: shows live metric values so thresholds can be tuned ---
            debug_lines = classifier.debug_lines()
            for i, line in enumerate(debug_lines):
                cv2.putText(bg, line, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(bg, f"mood: {mood}", (10, 25 + len(debug_lines) * 22 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            # --- END DEBUG OVERLAY ---q
        cv2.putText(bg, "q to quit", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(window_name, bg)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()