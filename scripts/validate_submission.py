"""
Pre-submission validator — run before zipping project for university.
  python scripts/validate_submission.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

FIGURES = [
    "fig01_dataset_class_distribution.png",
    "fig02_ml_metrics_comparison.png",
    "fig03_confusion_matrices_all_models.png",
    "fig04_f1_score_ranking.png",
    "fig05_best_model_knn_confusion_matrix.png",
    "fig06_flask_demo_me.png",
    "fig07_flask_demo_not_me.png",
]

REQUIRED_CODE = [
    ROOT / "app" / "flask_app.py",
    ROOT / "run_pipeline.py",
    ROOT / "src" / "preprocessing" / "preprocess_dataset.py",
    ROOT / "src" / "training" / "train_models.py",
    ROOT / "docs" / "REPORT.md",
    ROOT / "requirements.txt",
]

errors = []
warnings = []
ok = []


def check(path: Path, name: str, required=True):
    if path.exists():
        ok.append(f"OK  {name}")
    elif required:
        errors.append(f"MISSING (required): {name} -> {path}")
    else:
        warnings.append(f"Optional missing: {name}")


def main():
    print("=" * 60)
    print("SUBMISSION VALIDATOR")
    print("=" * 60)

    for p in REQUIRED_CODE:
        check(p, p.relative_to(ROOT).as_posix())

    check(config.BEST_MODEL_PATH, "Trained model (best_model.pkl)")
    check(config.METADATA_PATH, "model_metadata.json")
    check(config.PROCESSED_DATA_DIR / "X.npy", "Processed features X.npy")
    check(config.ME_SOURCE_DIR, "dataset/ME", required=False)
    check(config.NOT_ME_SOURCE_DIR, "dataset/NOT_ME", required=False)

    me_count = len(list(config.ME_SOURCE_DIR.glob("*"))) if config.ME_SOURCE_DIR.exists() else 0
    not_count = len(list(config.NOT_ME_SOURCE_DIR.glob("*"))) if config.NOT_ME_SOURCE_DIR.exists() else 0
    if me_count == 0:
        warnings.append("dataset/ME appears empty")
    else:
        ok.append(f"OK  dataset/ME ({me_count} items)")
    if not_count == 0:
        warnings.append("dataset/NOT_ME appears empty")
    else:
        ok.append(f"OK  dataset/NOT_ME ({not_count} items)")

    fig_dir = ROOT / "docs" / "figures"
    for f in FIGURES:
        check(fig_dir / f, f"Figure {f}", required=False)

    tm = config.TM_MODEL_PATH
    if tm.exists():
        ok.append("OK  Teachable Machine keras_model.h5")
    else:
        warnings.append("Teachable Machine model not exported (optional but recommended)")

    report = ROOT / "docs" / "REPORT.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        if "[Your Full Name]" in text:
            warnings.append("REPORT.md: Replace placeholder name on title page")
        if "93.10" in text or "93.1" in text:
            ok.append("OK  REPORT contains real metrics")

    if config.METADATA_PATH.exists():
        with open(config.METADATA_PATH) as f:
            meta = json.load(f)
        ok.append(f"OK  Best model: {meta.get('best_model')} (F1={meta.get('best_f1', 0):.4f})")

    print("\n--- Passed ---")
    for line in ok:
        print(line)
    if warnings:
        print("\n--- Warnings ---")
        for line in warnings:
            print(line)
    if errors:
        print("\n--- ERRORS ---")
        for line in errors:
            print(line)
        print("\nFix errors before submission.")
        return 1

    print("\n" + "=" * 60)
    print("READY FOR SUBMISSION (review warnings above)")
    print("Next: export docs/REPORT.md to PDF, zip project folder")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
