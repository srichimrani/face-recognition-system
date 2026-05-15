"""
Evaluation script: metrics tables, confusion matrix charts, model comparison plots.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config


def run_evaluation():
    """Generate charts and CSV tables from training metadata."""
    print("=" * 60)
    print("STEP 3: Evaluation & Visualization")
    print("=" * 60)

    if not config.METADATA_PATH.exists():
        raise FileNotFoundError("Train models first: python -m src.training.train_models")

    with open(config.METADATA_PATH) as f:
        meta = json.load(f)

    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results = meta["all_results"]
    df = pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "f1_score"]]
    df.to_csv(config.TABLES_DIR / "ml_model_comparison.csv", index=False)
    print(df.to_string(index=False))

    # Bar chart: metrics comparison
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Classical ML Model Comparison", fontsize=14, fontweight="bold")
    models = [r["model"] for r in results]
    colors = ["#2563eb", "#7c3aed", "#059669"]

    for ax, metric in zip(axes.flat, metrics):
        values = [r[metric] for r in results]
        bars = ax.bar(models, values, color=colors)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(config.CHARTS_DIR / "ml_metrics_comparison.png", dpi=150)
    plt.close()

    # Confusion matrices for each model
    from sklearn.metrics import ConfusionMatrixDisplay
    import joblib

    # Re-train quickly for CM plots only if we have processed data
    X = np.load(config.PROCESSED_DATA_DIR / "X.npy")
    y = np.load(config.PROCESSED_DATA_DIR / "y.npy")
    le = joblib.load(config.LABEL_ENCODER_PATH)
    y_enc = le.transform(y)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y_enc
    )
    scaler = joblib.load(config.SCALER_PATH)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    if config.PCA_PATH.exists():
        pca = joblib.load(config.PCA_PATH)
        X_train_s = pca.transform(X_train_s)
        X_test_s = pca.transform(X_test_s)

    clfs = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=config.RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=3),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, clf) in zip(axes, clfs.items()):
        clf.fit(X_train_s, y_train)
        ConfusionMatrixDisplay.from_predictions(
            y_test, clf.predict(X_test_s), display_labels=le.classes_, ax=ax, cmap="Blues"
        )
        ax.set_title(name)
    plt.suptitle("Confusion Matrices (Test Set)")
    plt.tight_layout()
    plt.savefig(config.CHARTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()

    print(f"\nCharts saved to {config.CHARTS_DIR}")
    print(f"Tables saved to {config.TABLES_DIR}")
    return meta


if __name__ == "__main__":
    run_evaluation()
