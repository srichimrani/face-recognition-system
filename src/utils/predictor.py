"""
Unified prediction interface for classical ML and Teachable Machine models.
"""
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

import config
from src.utils.face_detector import FaceDetector
from src.utils.feature_extractor import flatten_face


class ClassicalMLPredictor:
    """Load best sklearn model and predict ME / NOT ME from BGR image."""

    def __init__(self):
        if not config.BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Classical model not found. Run: python run_pipeline.py"
            )
        self.model = joblib.load(config.BEST_MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.le = joblib.load(config.LABEL_ENCODER_PATH)
        self.pca = joblib.load(config.PCA_PATH) if config.PCA_PATH.exists() else None
        self.detector = FaceDetector()
        self.model_name = "Classical ML"
        if config.METADATA_PATH.exists():
            with open(config.METADATA_PATH) as f:
                meta = json.load(f)
                self.model_name = meta.get("best_model", self.model_name)

    def predict(self, image_bgr: np.ndarray) -> dict:
        face, bbox = self.detector.detect_largest_face(image_bgr)
        if face is None:
            return {
                "success": False,
                "error": "No face detected. Ensure face is visible and well-lit.",
                "label": None,
                "confidence": 0.0,
                "model": self.model_name,
            }

        features = flatten_face(face).reshape(1, -1)
        features = self.scaler.transform(features)
        if self.pca is not None:
            features = self.pca.transform(features)

        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])
        else:
            pred_idx = int(self.model.predict(features)[0])
            confidence = 1.0

        label = self.le.inverse_transform([pred_idx])[0]
        annotated = self.detector.draw_bbox(image_bgr, bbox)

        return {
            "success": True,
            "label": label,
            "confidence": round(confidence * 100, 2),
            "is_me": label == config.CLASS_ME,
            "model": self.model_name,
            "annotated_image": annotated,
            "error": None,
        }


class TeachableMachinePredictor:
    """Predict using Keras model exported from Google Teachable Machine."""

    def __init__(self):
        self.detector = FaceDetector()
        self.model = None
        self.labels = []
        self.model_name = "Teachable Machine"
        self._load()

    def _find_model_file(self) -> Path | None:
        """Locate Keras .h5 from standard TM export path or folder search."""
        if config.TM_MODEL_PATH.exists():
            return config.TM_MODEL_PATH
        tm_dir = config.TM_MODEL_PATH.parent
        if not tm_dir.exists():
            return None
        for name in ("keras_model.h5", "model.h5", "converted_keras.h5"):
            p = tm_dir / name
            if p.exists():
                return p
        h5_files = list(tm_dir.glob("*.h5"))
        return h5_files[0] if h5_files else None

    def _load(self):
        model_path = self._find_model_file()
        if not model_path:
            return
        try:
            import tensorflow as tf

            self.model = tf.keras.models.load_model(str(model_path))
            if config.TM_LABELS_PATH.exists():
                with open(config.TM_LABELS_PATH, encoding="utf-8") as f:
                    self.labels = [line.strip().split(" ", 1)[-1] for line in f if line.strip()]
            else:
                # Infer class count from output layer
                n = int(self.model.output_shape[-1])
                self.labels = ["ME", "NOT ME"] if n == 2 else [str(i) for i in range(n)]
        except Exception as e:
            print(f"Teachable Machine load warning: {e}")
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def predict(self, image_bgr: np.ndarray) -> dict:
        if not self.is_available():
            return {
                "success": False,
                "error": "Teachable Machine model not found. Export keras_model.h5 to models/teachable_machine/",
                "label": None,
                "confidence": 0.0,
                "model": self.model_name,
            }

        face, bbox = self.detector.detect_largest_face(image_bgr)
        if face is None:
            return {
                "success": False,
                "error": "No face detected.",
                "label": None,
                "confidence": 0.0,
                "model": self.model_name,
            }

        # TM image model: RGB, size from input_shape (usually 224x224)
        ishape = self.model.input_shape
        if isinstance(ishape, list):
            ishape = ishape[0]
        h, w = int(ishape[1]), int(ishape[2])
        channels = int(ishape[3]) if len(ishape) > 3 and ishape[3] else 3

        if channels == 1:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (w, h))
            arr = np.expand_dims(resized, axis=-1)
        else:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (w, h))
            arr = resized

        arr = np.expand_dims(arr.astype(np.float32) / 255.0, axis=0)
        preds = self.model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        # Map TM class names to ME / NOT ME
        raw_label = self.labels[idx] if idx < len(self.labels) else str(idx)
        label = self._normalize_label(raw_label)
        annotated = self.detector.draw_bbox(image_bgr, bbox)

        return {
            "success": True,
            "label": label,
            "confidence": round(confidence * 100, 2),
            "is_me": label == config.CLASS_ME,
            "model": self.model_name,
            "annotated_image": annotated,
            "error": None,
        }

    @staticmethod
    def _normalize_label(raw: str) -> str:
        raw_upper = raw.upper().replace("_", " ")
        if "NOT" in raw_upper or raw_upper in ("0", "CLASS 2", "NOT ME", "NOT_ME"):
            return config.CLASS_NOT_ME
        return config.CLASS_ME
