#!/usr/bin/env python3
"""
Full ML pipeline: preprocess -> train -> evaluate.
Run from project root after adding images to data/raw/me and data/raw/not_me.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocess_dataset import run_preprocessing
from src.training.train_models import train_all_models
from src.evaluation.evaluate_models import run_evaluation


def main():
    print("\n" + "=" * 60)
    print("FACE RECOGNITION SYSTEM - TRAINING PIPELINE")
    print("=" * 60 + "\n")

    run_preprocessing()
    train_all_models()
    run_evaluation()

    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from generate_analysis import main as gen_analysis
        gen_analysis()
    except Exception as e:
        print(f"Note: Extended charts skipped ({e}). Run: python scripts/generate_analysis.py")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("Start web app: python app/flask_app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
