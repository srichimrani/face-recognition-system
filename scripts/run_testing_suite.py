"""
Automated testing suite for report metrics.
Place hold-out images in dataset/test/ME and dataset/test/NOT_ME.
"""
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.utils.predictor import ClassicalMLPredictor, TeachableMachinePredictor

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def run_tests():
    results = []
    ml = ClassicalMLPredictor()
    tm = TeachableMachinePredictor()
    tm_ok = tm.is_available()

    test_cases = [
        (config.CLASS_ME, config.TEST_ME_DIR),
        (config.CLASS_NOT_ME, config.TEST_NOT_ME_DIR),
    ]

    for true_label, folder in test_cases:
        if not folder.exists():
            print(f"Skip (not found): {folder}")
            continue
        for img_path in list_images(folder):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            ml_res = ml.predict(image)
            row = {
                "file": str(img_path.relative_to(folder)),
                "true_label": true_label,
                "ml_pred": ml_res.get("label"),
                "ml_correct": ml_res.get("label") == true_label if ml_res.get("success") else False,
                "ml_confidence": ml_res.get("confidence"),
            }
            if tm_ok:
                tm_res = tm.predict(image)
                row["tm_pred"] = tm_res.get("label")
                row["tm_correct"] = tm_res.get("label") == true_label if tm_res.get("success") else False
                row["tm_confidence"] = tm_res.get("confidence")
            results.append(row)

    if not results:
        print(
            "No test images found.\n"
            "Create:\n"
            f"  {config.TEST_ME_DIR}\n"
            f"  {config.TEST_NOT_ME_DIR}"
        )
        return

    df = pd.DataFrame(results)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TABLES_DIR / "live_testing_results.csv", index=False)

    summary = {"ml_accuracy": float(df["ml_correct"].mean())}
    if "tm_correct" in df.columns:
        summary["tm_accuracy"] = float(df["tm_correct"].mean())

    with open(config.TABLES_DIR / "live_testing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(df.to_string())
    print("\nSummary:", summary)
    print(f"Saved to {config.TABLES_DIR}")


if __name__ == "__main__":
    run_tests()
