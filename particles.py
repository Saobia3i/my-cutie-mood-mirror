"""
particles.py
Small, lightweight particle system for cute mood reactions.
Draws directly with OpenCV primitives -- no extra sprite assets needed.

- happy      -> little floating hearts
- surprised  -> star/spark burst
- sleepy     -> floating "z z z"
- neutral    -> no particles
"""

import random
import math
import cv2
import numpy as np


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "kind", "size", "color")

    def __init__(self, x, y, vx, vy, life, kind, size, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.kind = kind
        self.size = size
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy -= 0.05  # slight upward drift/gravity-lite
        self.life -= 1

    @property
    def alive(self):
        return self.life > 0

    @property
    def alpha(self):
        return max(0.0, self.life / self.max_life)


class ParticleSystem:
    def __init__(self, max_particles=120):
        self.particles = []
        self.max_particles = max_particles
        self._last_mood = None

    def _spawn_burst(self, cx, cy, kind, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1.5, 4.0)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 1.0
            life = random.randint(25, 45)
            size = random.randint(6, 14)

            if kind == "heart":
                color = random.choice([(147, 112, 219), (180, 130, 255), (255, 182, 193)])  # soft purples/pinks (BGR-ish)
            elif kind == "star":
                color = random.choice([(255, 220, 100), (255, 255, 150), (200, 255, 255)])
            else:  # zzz
                color = (230, 230, 230)

            self.particles.append(Particle(cx, cy, vx, vy, life, kind, size, color))

    def trigger(self, mood, cx, cy):
        """Call once when mood changes to spawn a burst appropriate to the new mood."""
        if mood == "happy":
            self._spawn_burst(cx, cy - 60, "heart", 14)
        elif mood == "surprised":
            self._spawn_burst(cx, cy - 60, "star", 18)
        elif mood == "sleepy":
            self._spawn_burst(cx, cy - 80, "zzz", 3)

    def maybe_ambient(self, mood, cx, cy, frame_count):
        """Light continuous trickle of particles while a mood is held (not just on change)."""
        if mood == "happy" and frame_count % 12 == 0:
            self._spawn_burst(cx, cy - 60, "heart", 2)
        elif mood == "sleepy" and frame_count % 30 == 0:
            self._spawn_burst(cx, cy - 80, "zzz", 1)

    def update_and_draw(self, frame_bgr):
        overlay = frame_bgr.copy()
        alive = []
        for p in self.particles:
            p.update()
            if not p.alive:
                continue
            alive.append(p)

            a = p.alpha
            color = tuple(int(c) for c in p.color)
            pt = (int(p.x), int(p.y))

            if p.kind == "heart":
                _draw_heart(overlay, pt, p.size, color, a)
            elif p.kind == "star":
                _draw_star(overlay, pt, p.size, color, a)
            elif p.kind == "zzz":
                cv2.putText(overlay, "z", pt, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6 + 0.4 * a, color, 2, cv2.LINE_AA)

        self.particles = alive[-self.max_particles:]
        cv2.addWeighted(overlay, 1.0, frame_bgr, 0.0, 0, dst=frame_bgr)
        return frame_bgr


def _draw_heart(img, center, size, color, alpha):
    x, y = center
    overlay = img.copy()
    r = max(2, size // 3)
    cv2.circle(overlay, (x - r, y - r), r, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + r, y - r), r, color, -1, cv2.LINE_AA)
    pts = np.array([[x - 2 * r, y - r], [x + 2 * r, y - r], [x, y + 2 * r]], np.int32)
    cv2.fillConvexPoly(overlay, pts, color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def _draw_star(img, center, size, color, alpha):
    x, y = center
    overlay = img.copy()
    pts = []
    for i in range(4):
        ang = i * math.pi / 2
        pts.append((int(x + size * math.cos(ang)), int(y + size * math.sin(ang))))
        ang2 = ang + math.pi / 4
        pts.append((int(x + size * 0.35 * math.cos(ang2)), int(y + size * 0.35 * math.sin(ang2))))
    pts = np.array(pts, np.int32)
    cv2.fillConvexPoly(overlay, pts, color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
