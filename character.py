"""
character.py
Loads the 4 mood sprites and renders the character centered on the frame,
with a soft crossfade between mood changes and a gentle idle "breathing" bounce.
"""

import os
import time
import math
import cv2
import numpy as np


STATES = ["neutral", "happy", "surprised", "sleepy", "angry", "sad", "tongue"]


def _overlay_rgba(background_bgr, sprite_rgba, center_x, center_y, scale=1.0, alpha_mult=1.0):
    """
    Alpha-blends an RGBA sprite onto a BGR background, centered at (center_x, center_y).
    scale resizes the sprite. alpha_mult (0-1) fades the whole sprite in/out.
    """
    sh, sw = sprite_rgba.shape[:2]
    new_w, new_h = int(sw * scale), int(sh * scale)
    if new_w <= 0 or new_h <= 0:
        return background_bgr

    sprite_resized = cv2.resize(sprite_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x1 = center_x - new_w // 2
    y1 = center_y - new_h // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    bh, bw = background_bgr.shape[:2]

    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2, sy2 = new_w - max(0, x2 - bw), new_h - max(0, y2 - bh)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(bw, x2), min(bh, y2)

    if x2c <= x1c or y2c <= y1c:
        return background_bgr

    sprite_crop = sprite_resized[sy1:sy2, sx1:sx2]
    region = background_bgr[y1c:y2c, x1c:x2c]

    alpha = (sprite_crop[:, :, 3] / 255.0) * alpha_mult
    alpha = alpha[:, :, np.newaxis]
    fg = sprite_crop[:, :, :3].astype(float)
    bg = region.astype(float)

    blended = fg * alpha + bg * (1 - alpha)
    background_bgr[y1c:y2c, x1c:x2c] = blended.astype(np.uint8)
    return background_bgr


class Character:
    def __init__(self, assets_dir, base_size=320, transition_duration=0.35):
        self.sprites = {}
        for state in STATES:
            path = os.path.join(assets_dir, f"{state}.png")
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Missing sprite: {path}")
            if img.shape[2] == 3:
                b, g, r = cv2.split(img)
                a = np.ones(b.shape, dtype=b.dtype) * 255
                img = cv2.merge((b, g, r, a))
            self.sprites[state] = img

        self.base_size = base_size
        self.current_state = "neutral"
        self.target_state = "neutral"
        self.transition = 1.0
        self.transition_duration = transition_duration
        self._transition_start_time = time.time()
        self._start_time = time.time()

    def set_mood(self, new_state):
        if new_state not in self.sprites:
            new_state = "neutral"
        if new_state != self.target_state:
            self.target_state = new_state
            self.transition = 0.0
            self._transition_start_time = time.time()

    def _idle_bounce_offset(self):
        t = time.time() - self._start_time
        return int(6 * math.sin(t * 2.0))

    def render(self, frame_bgr, center_x=None, center_y=None):
        h, w = frame_bgr.shape[:2]
        if center_x is None:
            center_x = w // 2
        if center_y is None:
            center_y = h // 2

        bounce = self._idle_bounce_offset()
        cy = center_y + bounce

        pop_scale = 1.0
        if self.transition < 1.0:
            pop_scale = 1.0 + 0.08 * math.sin(self.transition * math.pi)

        if self.transition >= 1.0:
            sprite = self.sprites[self.target_state]
            frame_bgr = _overlay_rgba(frame_bgr, sprite, center_x, cy,
                                       scale=(self.base_size / sprite.shape[1]) * pop_scale)
            self.current_state = self.target_state
        else:
            old_sprite = self.sprites[self.current_state]
            new_sprite = self.sprites[self.target_state]
            frame_bgr = _overlay_rgba(frame_bgr, old_sprite, center_x, cy,
                                       scale=(self.base_size / old_sprite.shape[1]) * pop_scale,
                                       alpha_mult=1.0 - self.transition)
            frame_bgr = _overlay_rgba(frame_bgr, new_sprite, center_x, cy,
                                       scale=(self.base_size / new_sprite.shape[1]) * pop_scale,
                                       alpha_mult=self.transition)
            elapsed = time.time() - self._transition_start_time
            self.transition = min(1.0, elapsed / self.transition_duration)

        return frame_bgr, (center_x, cy)