"""
Generate charts, comparison tables, and analysis JSON from trained models.
Run after run_pipeline.py:  python scripts/generate_analysis.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

sns.set_theme(style="whitegrid", font_scale=1.05)


def load_meta():
    with open(config.METADATA_PATH) as f:
        return json.load(f)


def load_preprocess_meta():
    p = config.PROCESSED_DATA_DIR / "preprocessing_metadata.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def chart_dataset_balance(pre_meta: dict):
    labels = ["ME", "NOT ME"]
    counts = [pre_meta.get("n_me", 0), pre_meta.get("n_not_me", 0)]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=["#22c55e", "#ef4444"])
    ax.set_title("Dataset Class Distribution (After Face Detection)")
    ax.set_ylabel("Number of Images")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 1, str(c), ha="center")
    plt.tight_layout()
    plt.savefig(config.CHARTS_DIR / "dataset_class_distribution.png", dpi=150)
    plt.close()


def chart_f1_comparison(meta: dict):
    df = pd.DataFrame(meta["all_results"])
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2563eb", "#7c3aed", "#059669"]
    bars = ax.barh(df["model"], df["f1_score"], color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-Score (Weighted)")
    ax.set_title("Model Selection — F1-Score Comparison")
    ax.axvline(meta["best_f1"], color="#f59e0b", linestyle="--", label=f"Best: {meta['best_model']}")
    for bar, v in zip(bars, df["f1_score"]):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center")
    ax.legend()
    plt.tight_layout()
    plt.savefig(config.CHARTS_DIR / "f1_score_ranking.png", dpi=150)
    plt.close()


def chart_best_confusion(meta: dict):
    best = next(r for r in meta["all_results"] if r["model"] == meta["best_model"])
    cm = np.array(best["confusion_matrix"])
    classes = meta.get("classes", ["ME", "NOT ME"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Best Model ({meta['best_model']}) — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(config.CHARTS_DIR / "best_model_confusion_matrix.png", dpi=150)
    plt.close()


def write_performance_tables(meta: dict, pre_meta: dict):
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(meta["all_results"])
    df["accuracy_pct"] = (df["accuracy"] * 100).round(2)
    df["f1_pct"] = (df["f1_score"] * 100).round(2)
    df.to_csv(config.TABLES_DIR / "ml_model_comparison.csv", index=False)

    summary = {
        "project": "Web-Based Visual Recognition System",
        "dataset_total": pre_meta.get("n_samples", 0),
        "dataset_me": pre_meta.get("n_me", 0),
        "dataset_not_me": pre_meta.get("n_not_me", 0),
        "train_samples": meta.get("train_samples"),
        "test_samples": meta.get("test_samples"),
        "best_model": meta.get("best_model"),
        "best_f1": round(meta.get("best_f1", 0), 4),
        "best_accuracy": next(
            r["accuracy"] for r in meta["all_results"] if r["model"] == meta["best_model"]
        ),
        "split": "80/20 stratified",
        "feature_dim_raw": pre_meta.get("feature_dim"),
        "pca_components": config.PCA_COMPONENTS if config.USE_PCA else None,
    }
    with open(config.TABLES_DIR / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# Performance Results (Auto-Generated)\n",
        f"**Generated from trained models** | Best model: **{meta['best_model']}**\n",
        "## Dataset Summary\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total processed faces | {pre_meta.get('n_samples', 'N/A')} |",
        f"| ME class | {pre_meta.get('n_me', 'N/A')} |",
        f"| NOT ME class | {pre_meta.get('n_not_me', 'N/A')} |",
        f"| Train / Test split | {meta.get('train_samples')} / {meta.get('test_samples')} |",
        "",
        "## Classical ML Comparison (Test Set)\n",
        "| Model | Accuracy | Precision | Recall | F1-Score |",
        "|-------|----------|-----------|--------|----------|",
    ]
    for r in meta["all_results"]:
        md_lines.append(
            f"| {r['model']} | {r['accuracy']*100:.2f}% | {r['precision']*100:.2f}% | "
            f"{r['recall']*100:.2f}% | {r['f1_score']*100:.2f}% |"
        )

    best = next(r for r in meta["all_results"] if r["model"] == meta["best_model"])
    cm = best["confusion_matrix"]
    classes = meta.get("classes", ["ME", "NOT ME"])
    md_lines.extend([
        "",
        f"## Best Model Confusion Matrix ({meta['best_model']})\n",
        f"Classes order: {classes}\n",
        "| | Pred " + classes[0] + " | Pred " + classes[1] + " |",
        "|---|---|---|",
        f"| Actual {classes[0]} | {cm[0][0]} | {cm[0][1]} |",
        f"| Actual {classes[1]} | {cm[1][0]} | {cm[1][1]} |",
        "",
        "## Charts\n",
        "- `results/charts/ml_metrics_comparison.png`",
        "- `results/charts/confusion_matrices.png`",
        "- `results/charts/dataset_class_distribution.png`",
        "- `results/charts/f1_score_ranking.png`",
        "- `results/charts/best_model_confusion_matrix.png`",
    ])

    out = config.TABLES_DIR / "PERFORMANCE_RESULTS.md"
    out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out}")


def main():
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    pre = load_preprocess_meta()

    from src.evaluation.evaluate_models import run_evaluation

    run_evaluation()
    chart_dataset_balance(pre)
    chart_f1_comparison(meta)
    chart_best_confusion(meta)
    write_performance_tables(meta, pre)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
