"""
Compare Classical ML (kNN) vs Teachable Machine on dataset images.
Requires: run_pipeline.py + TM export in models/teachable_machine/

  python scripts/compare_ml_tm.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.utils.predictor import ClassicalMLPredictor, TeachableMachinePredictor

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_samples():
    rows = []
    for folder, label in [(config.ME_SOURCE_DIR, config.CLASS_ME), (config.NOT_ME_SOURCE_DIR, config.CLASS_NOT_ME)]:
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in EXTENSIONS:
                rows.append({"path": p, "label": label})
    return rows


def evaluate_predictor(predictor, samples, name: str):
    y_true, y_pred, confidences = [], [], []
    errors = 0
    for s in samples:
        img = cv2.imread(str(s["path"]))
        if img is None:
            errors += 1
            continue
        res = predictor.predict(img)
        if not res.get("success"):
            errors += 1
            continue
        y_true.append(s["label"])
        y_pred.append(res["label"])
        confidences.append(res.get("confidence", 0))

    if not y_true:
        return None

    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "n_evaluated": len(y_true),
        "n_errors": errors,
        "avg_confidence": float(np.mean(confidences)) if confidences else 0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[config.CLASS_ME, config.CLASS_NOT_ME]).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def main():
    print("=" * 60)
    print("ML vs Teachable Machine Comparison")
    print("=" * 60)

    samples = collect_samples()
    if len(samples) < 4:
        print("Not enough images in dataset/ME and dataset/NOT_ME")
        return 1

    _, test_samples = train_test_split(
        samples,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=[s["label"] for s in samples],
    )
    print(f"Test samples: {len(test_samples)} (same split seed as training)\n")

    ml = ClassicalMLPredictor()
    tm = TeachableMachinePredictor()

    if not tm.is_available():
        print("Teachable Machine not loaded.")
        print("Run: python scripts/setup_teachable_machine.py <your-export.zip>")
        return 1

    ml_res = evaluate_predictor(ml, test_samples, f"Classical ML ({ml.model_name})")
    tm_res = evaluate_predictor(tm, test_samples, "Teachable Machine")

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in (ml_res, tm_res):
        rows.append(
            {
                "model": r["model"],
                "accuracy": round(r["accuracy"], 4),
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1_score": round(r["f1_score"], 4),
                "avg_confidence": round(r["avg_confidence"], 2),
                "n_evaluated": r["n_evaluated"],
            }
        )
        print(f"\n{r['model']}")
        print(f"  Accuracy:  {r['accuracy']*100:.2f}%")
        print(f"  F1-Score:  {r['f1_score']*100:.2f}%")
        print(f"  Avg conf:  {r['avg_confidence']:.2f}%")
        print(r["report"])

    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES_DIR / "ml_vs_tm_comparison.csv", index=False)

    comparison = {"ml": ml_res, "teachable_machine": tm_res}
    with open(config.TABLES_DIR / "ml_vs_tm_comparison.json", "w") as f:
        json.dump(
            {
                "ml": {k: v for k, v in ml_res.items() if k != "report"},
                "tm": {k: v for k, v in tm_res.items() if k != "report"},
            },
            f,
            indent=2,
        )

    # Comparison bar chart
    import matplotlib.pyplot as plt

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    w = 0.35
    ml_vals = [ml_res[m] for m in metrics]
    tm_vals = [tm_res[m] for m in metrics]
    ax.bar(x - w / 2, ml_vals, w, label=ml_res["model"], color="#2563eb")
    ax.bar(x + w / 2, tm_vals, w, label=tm_res["model"], color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_title("Classical ML vs Teachable Machine (Same Test Split)")
    ax.legend()
    for i, (a, b) in enumerate(zip(ml_vals, tm_vals)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(config.CHARTS_DIR / "ml_vs_tm_comparison.png", dpi=150)
    plt.close()

    md = [
        "# ML vs Teachable Machine Comparison\n",
        f"Test samples: **{len(test_samples)}** | Split: 20% stratified, seed={config.RANDOM_STATE}\n",
        "| Model | Accuracy | Precision | Recall | F1 | Avg Confidence |",
        "|-------|----------|-----------|--------|-----|----------------|",
    ]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['accuracy']*100:.2f}% | {r['precision']*100:.2f}% | "
            f"{r['recall']*100:.2f}% | {r['f1_score']*100:.2f}% | {r['avg_confidence']}% |"
        )
    (config.TABLES_DIR / "ML_VS_TM_RESULTS.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nSaved: {config.TABLES_DIR / 'ml_vs_tm_comparison.csv'}")
    print(f"Chart: {config.CHARTS_DIR / 'ml_vs_tm_comparison.png'}")
    print("Add results to REPORT.md Section 9.3 and 10.2")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
