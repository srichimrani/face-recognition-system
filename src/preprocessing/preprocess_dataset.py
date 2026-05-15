"""
Dataset preprocessing pipeline:
1. Load images from dataset/ME and dataset/NOT_ME
2. Face detection (Haar Cascade)
3. Crop and resize faces
4. Grayscale conversion and normalization
5. Feature extraction (flattened pixels)
6. Save processed arrays and metadata for training
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from src.utils.face_detector import FaceDetector
from src.utils.feature_extractor import flatten_face

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    """All images in folder (including subfolders)."""
    if not folder.exists():
        return []
    files = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return files


def load_images_from_folder(folder: Path, label: str, save_subdir: Path) -> list[dict]:
    """Load images, detect faces, extract features."""
    records = []
    detector = FaceDetector()
    save_subdir.mkdir(parents=True, exist_ok=True)

    files = list_images(folder)
    if not files:
        print(f"Warning: no images in {folder}")

    for img_path in files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Skipped (unreadable): {img_path.relative_to(folder)}")
            continue

        face, bbox = detector.detect_largest_face(image)
        if face is None:
            print(f"  Skipped (no face): {img_path.name}")
            continue

        features = flatten_face(face)
        out_name = f"{img_path.stem}_{img_path.parent.name}_{img_path.suffix}"
        out_name = out_name.replace(" ", "_")
        cv2.imwrite(str(save_subdir / out_name), face)

        records.append(
            {
                "source": str(img_path.relative_to(folder)),
                "label": label,
                "features": features.tolist(),
            }
        )
        print(f"  OK: {img_path.name} -> {label}")

    return records


def run_preprocessing() -> dict:
    """Run full preprocessing and save X, y arrays."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    me_dir, not_me_dir = config.get_training_dirs()

    print("=" * 60)
    print("STEP 1: Face Detection & Feature Extraction")
    print("=" * 60)
    print(f"ME folder:      {me_dir}")
    print(f"NOT_ME folder:  {not_me_dir}")

    if not me_dir.exists():
        raise FileNotFoundError(
            f"ME folder not found: {me_dir}\n"
            f"Create dataset/ME/ and add your face images."
        )
    if not not_me_dir.exists():
        raise FileNotFoundError(
            f"NOT_ME folder not found: {not_me_dir}\n"
            f"Create dataset/NOT_ME/ and add other people's images."
        )

    me_records = load_images_from_folder(me_dir, config.CLASS_ME, config.PROCESSED_ME_DIR)
    not_me_records = load_images_from_folder(
        not_me_dir, config.CLASS_NOT_ME, config.PROCESSED_NOT_ME_DIR
    )
    all_records = me_records + not_me_records

    if len(me_records) == 0 or len(not_me_records) == 0:
        raise RuntimeError(
            f"Need images in BOTH classes. Found ME={len(me_records)}, NOT_ME={len(not_me_records)}."
        )
    if len(all_records) < 4:
        raise RuntimeError(
            "Not enough processed samples. Add at least 10 images per class in "
            "dataset/ME and dataset/NOT_ME, then run again."
        )

    X = np.array([r["features"] for r in all_records], dtype=np.float32)
    y = np.array([r["label"] for r in all_records])

    np.save(config.PROCESSED_DATA_DIR / "X.npy", X)
    np.save(config.PROCESSED_DATA_DIR / "y.npy", y)

    meta = {
        "dataset_structure": "dataset/ME, dataset/NOT_ME",
        "me_source": str(me_dir),
        "not_me_source": str(not_me_dir),
        "n_samples": len(all_records),
        "n_me": len(me_records),
        "n_not_me": len(not_me_records),
        "feature_dim": int(X.shape[1]),
        "face_size": config.FACE_SIZE,
        "samples": [{"source": r["source"], "label": r["label"]} for r in all_records],
    }
    with open(config.PROCESSED_DATA_DIR / "preprocessing_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nProcessed {len(all_records)} samples (ME: {len(me_records)}, NOT ME: {len(not_me_records)})")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Cropped faces saved under {config.PROCESSED_DATA_DIR}")
    return meta


if __name__ == "__main__":
    run_preprocessing()
