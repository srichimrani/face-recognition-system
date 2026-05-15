"""
Global configuration for the Face Recognition System.
Adjust paths and hyperparameters here before training or running the app.
"""
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Primary dataset (put your images here)
#   dataset/ME/       → your face
#   dataset/NOT_ME/   → other people
# ---------------------------------------------------------------------------
DATASET_DIR = BASE_DIR / "dataset"
ME_SOURCE_DIR = DATASET_DIR / "ME"
NOT_ME_SOURCE_DIR = DATASET_DIR / "NOT_ME"

# Legacy / optional mirror (populated by scripts/import_dataset.py if needed)
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
ME_DIR = RAW_DATA_DIR / "me"
NOT_ME_DIR = RAW_DATA_DIR / "not_me"

# Read training images directly from dataset/ME and dataset/NOT_ME
USE_DATASET_FOLDER = True

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_ME_DIR = PROCESSED_DATA_DIR / "ME"
PROCESSED_NOT_ME_DIR = PROCESSED_DATA_DIR / "NOT_ME"

# Hold-out test images (optional): dataset/test/ME and dataset/test/NOT_ME
TEST_DATASET_DIR = DATASET_DIR / "test"
TEST_ME_DIR = TEST_DATASET_DIR / "ME"
TEST_NOT_ME_DIR = TEST_DATASET_DIR / "NOT_ME"

# Model and results paths
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
PCA_PATH = MODELS_DIR / "pca.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Teachable Machine export (Keras .h5 from Google Teachable Machine)
TM_MODEL_PATH = MODELS_DIR / "teachable_machine" / "keras_model.h5"
TM_LABELS_PATH = MODELS_DIR / "teachable_machine" / "labels.txt"

# Results output
RESULTS_DIR = BASE_DIR / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
TABLES_DIR = RESULTS_DIR / "tables"

# Face detection and preprocessing
FACE_SIZE = (100, 100)  # Width x Height after crop
GRAYSCALE = True
TEST_SIZE = 0.2
RANDOM_STATE = 42

# PCA for dimensionality reduction on flattened face pixels
USE_PCA = True
PCA_COMPONENTS = 50

# Class labels (display names in UI and report)
CLASS_ME = "ME"
CLASS_NOT_ME = "NOT ME"
CLASS_LABELS = [CLASS_ME, CLASS_NOT_ME]

# Flask settings
SECRET_KEY = "face-recognition-university-project-2026"
DEBUG = True
HOST = "127.0.0.1"
PORT = 5000

CONFIDENCE_THRESHOLD = 0.55


def get_training_dirs() -> tuple[Path, Path]:
    """Return (ME folder, NOT_ME folder) for training/preprocessing."""
    if USE_DATASET_FOLDER:
        return ME_SOURCE_DIR, NOT_ME_SOURCE_DIR
    return ME_DIR, NOT_ME_DIR
