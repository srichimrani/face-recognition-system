"""
Face detection using OpenCV Haar Cascade classifier.
Detects the largest face in an image and returns a cropped, resized region.
"""
import cv2
import numpy as np
from pathlib import Path

import config


class FaceDetector:
    """Detects faces in images using OpenCV's pre-trained Haar cascade."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("Failed to load Haar cascade for face detection.")

    def detect_largest_face(self, image: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
        """
        Find the largest face in the image.

        Returns:
            (cropped_face, bbox) or (None, None) if no face detected.
            bbox is (x, y, w, h).
        """
        if image is None or image.size == 0:
            return None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return None, None

        # Select largest face by area
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = image[y : y + h, x : x + w]
        face_resized = cv2.resize(face, config.FACE_SIZE)
        return face_resized, (int(x), int(y), int(w), int(h))

    def draw_bbox(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        """Draw bounding box on image for visualization."""
        if bbox is None:
            return image
        x, y, w, h = bbox
        out = image.copy()
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return out
