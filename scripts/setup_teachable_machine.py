"""
Install Teachable Machine export into the project.

Usage:
  python scripts/setup_teachable_machine.py "C:\Downloads\converted_keras.zip"
  python scripts/setup_teachable_machine.py "C:\path\to\extracted_folder"

Or manually copy keras_model.h5 + labels.txt to models/teachable_machine/
Then run: python scripts/setup_teachable_machine.py --verify
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

TM_DIR = config.TM_MODEL_PATH.parent
H5_NAMES = ("keras_model.h5", "model.h5", "converted_keras.h5")
LABEL_NAMES = ("labels.txt", "labels.csv")


def find_in_dir(folder: Path) -> tuple[Path | None, Path | None]:
    h5, labels = None, None
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.name in H5_NAMES or (p.suffix == ".h5" and "model" in p.name.lower()):
            h5 = p
        if p.name in LABEL_NAMES:
            labels = p
    if h5 is None:
        for p in folder.rglob("*.h5"):
            h5 = p
            break
    return h5, labels


def install_from_path(source: Path):
    TM_DIR.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() == ".zip":
        extract_to = TM_DIR / "_extract_tmp"
        if extract_to.exists():
            shutil.rmtree(extract_to)
        extract_to.mkdir()
        with zipfile.ZipFile(source, "r") as zf:
            zf.extractall(extract_to)
        source = extract_to

    h5, labels = find_in_dir(source)
    if not h5:
        print("ERROR: No .h5 model found in:", source)
        print("Export from Teachable Machine: TensorFlow -> Keras")
        return False

    shutil.copy2(h5, config.TM_MODEL_PATH)
    print(f"Model -> {config.TM_MODEL_PATH}")

    if labels:
        shutil.copy2(labels, config.TM_LABELS_PATH)
        print(f"Labels -> {config.TM_LABELS_PATH}")
    else:
        default_labels = "0 ME\n1 NOT ME\n"
        config.TM_LABELS_PATH.write_text(default_labels, encoding="utf-8")
        print(f"Created default {config.TM_LABELS_PATH}")

    tmp = TM_DIR / "_extract_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    return True


def verify():
    from src.utils.predictor import TeachableMachinePredictor

    p = TeachableMachinePredictor()
    if not p.is_available():
        print("FAILED: Model did not load.")
        print(f"Expected: {config.TM_MODEL_PATH}")
        return False

    import tensorflow as tf

    m = p.model
    print("OK — Teachable Machine loaded")
    print(f"  Input shape: {m.input_shape}")
    print(f"  Output shape: {m.output_shape}")
    print(f"  Labels: {p.labels}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Install TM Keras export")
    parser.add_argument("source", nargs="?", help="ZIP or folder from TM export")
    parser.add_argument("--verify", action="store_true", help="Only verify load")
    args = parser.parse_args()

    if args.verify or not args.source:
        if not config.TM_MODEL_PATH.exists():
            print("Model not installed yet.")
            print("\nSteps:")
            print("  1. Train at https://teachablemachine.withgoogle.com/")
            print("  2. Export -> TensorFlow -> Keras -> Download ZIP")
            print("  3. python scripts/setup_teachable_machine.py <path-to-zip>")
            return 1
        return 0 if verify() else 1

    source = Path(args.source)
    if not source.exists():
        print(f"Not found: {source}")
        return 1

    if install_from_path(source):
        return 0 if verify() else 1
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
