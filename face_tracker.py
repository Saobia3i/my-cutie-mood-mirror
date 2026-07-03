"""
face_tracker.py
Uses MediaPipe's newer Tasks API (FaceLandmarker) instead of the deprecated
legacy `mp.solutions` module, which was removed from recent mediapipe pip
releases (0.10.30+). Landmark indices/topology are the same 478-point mesh
as before, so the rest of the app (expression_classifier.py) needs no changes.

On first run this downloads a small (~4MB) face landmarker model file into
the project folder -- needs an internet connection once.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model (one-time, ~4MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)


class FaceTracker:
    def __init__(self, max_faces=1):
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr):
        """
        Runs face landmark detection on a BGR frame.
        Returns a list of (x, y) pixel-space landmarks for the first detected face,
        or None if no face is found.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._timestamp_ms += 33  # timestamps must strictly increase between calls
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in face]
        return points

    def close(self):
        self.landmarker.close()