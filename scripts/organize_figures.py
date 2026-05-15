"""
Copy generated charts and user screenshots into docs/figures/ for the report.
Usage:
  1. Save Flask screenshots as docs/figures/incoming/flask_me.png and flask_not_me.png
  2. Run: python scripts/organize_figures.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "results" / "charts"
INCOMING = ROOT / "docs" / "figures" / "incoming"
OUT = ROOT / "docs" / "figures"

MAPPING = {
    "dataset_class_distribution.png": "fig01_dataset_class_distribution.png",
    "ml_metrics_comparison.png": "fig02_ml_metrics_comparison.png",
    "confusion_matrices.png": "fig03_confusion_matrices_all_models.png",
    "f1_score_ranking.png": "fig04_f1_score_ranking.png",
    "best_model_confusion_matrix.png": "fig05_best_model_knn_confusion_matrix.png",
    "ml_vs_tm_comparison.png": "fig08_ml_vs_tm_comparison.png",
}

SCREENSHOTS = {
    "flask_me.png": "fig06_flask_demo_me.png",
    "flask_not_me.png": "fig07_flask_demo_not_me.png",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    INCOMING.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_name, dst_name in MAPPING.items():
        src = CHARTS / src_name
        if src.exists():
            shutil.copy2(src, OUT / dst_name)
            print(f"Chart: {dst_name}")
            copied += 1
        else:
            print(f"Missing chart: {src}")

    for src_name, dst_name in SCREENSHOTS.items():
        for folder in (INCOMING, OUT, ROOT):
            src = folder / src_name
            if src.exists() and folder != OUT:
                shutil.copy2(src, OUT / dst_name)
                print(f"Screenshot: {dst_name}")
                copied += 1
                break
        else:
            if not (OUT / dst_name).exists():
                print(f"Add screenshot: {INCOMING / src_name}")

    print(f"\nFigures folder: {OUT}")
    print(f"Copied/verified {copied} files. See docs/APPENDIX_FIGURES.md")


if __name__ == "__main__":
    main()
