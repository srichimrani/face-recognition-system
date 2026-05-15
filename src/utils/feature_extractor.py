"""
Feature extraction from face images.
Converts cropped faces into normalized feature vectors for classical ML.
"""
import cv2
import numpy as np

import config


def flatten_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Resize is assumed done. Convert to grayscale, normalize to [0,1], flatten.
    """
    if face_bgr is None:
        return np.array([])

    if config.GRAYSCALE:
        if len(face_bgr.shape) == 3:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_bgr
    else:
        gray = face_bgr

    # Normalization: pixel values 0-255 -> 0.0-1.0 (standard for ML)
    normalized = gray.astype(np.float32) / 255.0
    return normalized.flatten()


def extract_features_from_image(image_bgr: np.ndarray, detector) -> np.ndarray | None:
    """Detect face and return feature vector, or None if no face."""
    face, _ = detector.detect_largest_face(image_bgr)
    if face is None:
        return None
    return flatten_face(face)
