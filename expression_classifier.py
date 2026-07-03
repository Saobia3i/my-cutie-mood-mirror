"""
expression_classifier.py
Lightweight, rule-based mood detection from MediaPipe Face Mesh landmarks.
No deep learning needed -- just landmark distance ratios.

MediaPipe Face Mesh key indices used here:
  Mouth:        left corner=61   right corner=291   top lip=13   bottom lip=14
  Left eye:     33, 160, 158, 133, 153, 144   (EAR points)
  Right eye:    362, 385, 387, 263, 373, 380  (EAR points)
  Eyebrows:     left=105   right=334
  Face width ref: left cheek=234   right cheek=454   (for scale normalization)
"""

import math
from collections import deque


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _eye_aspect_ratio(points, eye_idx):
    p = [points[i] for i in eye_idx]
    vertical = (_dist(p[1], p[5]) + _dist(p[2], p[4])) / 2.0
    horizontal = _dist(p[0], p[3])
    if horizontal == 0:
        return 0
    return vertical / horizontal


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_BROW, LEFT_EYE_TOP = 105, 159
RIGHT_BROW, RIGHT_EYE_TOP = 334, 386


class ExpressionClassifier:
    """
    Call update(landmarks) every frame with the pixel-space landmark list
    from FaceTracker. Returns one of:
        'neutral', 'happy', 'surprised', 'sleepy', 'angry', 'sad'.

    Internally smooths the raw metrics over a short window so the mood
    doesn't flicker frame-to-frame.
    """

    def __init__(self, smoothing_window=4):
        self.smoothing_window = smoothing_window
        self.mouth_open_hist = deque(maxlen=smoothing_window)
        self.smile_hist = deque(maxlen=smoothing_window)
        self.ear_hist = deque(maxlen=smoothing_window)
        self.brow_hist = deque(maxlen=smoothing_window)
        self.frown_hist = deque(maxlen=smoothing_window)

        self.EAR_SLEEPY_THRESH = 0.16
        self.MOUTH_OPEN_SURPRISE = 0.10
        self.SMILE_RATIO_THRESH = 0.42
        self.BROW_ANGRY_THRESH = 0.18
        self.FROWN_SAD_THRESH = 0.05

        self.last_metrics = {"mouth_open": 0, "smile": 0, "ear": 0, "brow": 0, "frown": 0}

    def debug_lines(self):
        m = self.last_metrics
        return [
            f"mouth_open: {m['mouth_open']:.3f}  (surprise > {self.MOUTH_OPEN_SURPRISE})",
            f"smile:      {m['smile']:.3f}  (happy > {self.SMILE_RATIO_THRESH})",
            f"ear:        {m['ear']:.3f}  (sleepy < {self.EAR_SLEEPY_THRESH})",
            f"brow:       {m['brow']:.3f}  (angry < {self.BROW_ANGRY_THRESH})",
            f"frown:      {m['frown']:.3f}  (sad > {self.FROWN_SAD_THRESH})",
        ]

    def update(self, landmarks):
        if landmarks is None:
            return "neutral"

        face_width = _dist(landmarks[234], landmarks[454])
        if face_width == 0:
            return "neutral"

        mouth_width = _dist(landmarks[61], landmarks[291])
        mouth_height = _dist(landmarks[13], landmarks[14])

        mouth_open_ratio = mouth_height / face_width
        smile_ratio = mouth_width / face_width

        left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        brow_dist = (_dist(landmarks[LEFT_BROW], landmarks[LEFT_EYE_TOP]) +
                     _dist(landmarks[RIGHT_BROW], landmarks[RIGHT_EYE_TOP])) / 2.0
        brow_ratio = brow_dist / face_width

        mouth_corner_avg_y = (landmarks[61][1] + landmarks[291][1]) / 2.0
        frown_amount = (mouth_corner_avg_y - landmarks[13][1]) / face_width

        self.mouth_open_hist.append(mouth_open_ratio)
        self.smile_hist.append(smile_ratio)
        self.ear_hist.append(avg_ear)
        self.brow_hist.append(brow_ratio)
        self.frown_hist.append(frown_amount)

        smooth_mouth_open = sum(self.mouth_open_hist) / len(self.mouth_open_hist)
        smooth_smile = sum(self.smile_hist) / len(self.smile_hist)
        smooth_ear = sum(self.ear_hist) / len(self.ear_hist)
        smooth_brow = sum(self.brow_hist) / len(self.brow_hist)
        smooth_frown = sum(self.frown_hist) / len(self.frown_hist)

        self.last_metrics = {
            "mouth_open": smooth_mouth_open,
            "smile": smooth_smile,
            "ear": smooth_ear,
            "brow": smooth_brow,
            "frown": smooth_frown,
        }

        if smooth_ear < self.EAR_SLEEPY_THRESH:
            return "sleepy"
        elif smooth_mouth_open > self.MOUTH_OPEN_SURPRISE:
            return "surprised"
        elif smooth_brow < self.BROW_ANGRY_THRESH and smooth_smile < self.SMILE_RATIO_THRESH:
            return "angry"
        elif smooth_smile > self.SMILE_RATIO_THRESH:
            return "happy"
        elif smooth_frown > self.FROWN_SAD_THRESH:
            return "sad"
        else:
            return "neutral"