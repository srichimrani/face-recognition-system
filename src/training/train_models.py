"""
Train and compare classical ML models:
- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (kNN)

Saves the best model by F1-score on the test set.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config


def load_processed_data():
    X_path = config.PROCESSED_DATA_DIR / "X.npy"
    y_path = config.PROCESSED_DATA_DIR / "y.npy"
    if not X_path.exists():
        raise FileNotFoundError("Run preprocessing first: python -m src.preprocessing.preprocess_dataset")
    return np.load(X_path), np.load(y_path)


def train_all_models() -> dict:
    """Train all models, save best, return comparison metrics."""
    print("=" * 60)
    print("STEP 2: Training Classical ML Models")
    print("=" * 60)

    X, y = load_processed_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = None
    if config.USE_PCA and X_train_s.shape[1] > config.PCA_COMPONENTS:
        n_comp = min(config.PCA_COMPONENTS, X_train_s.shape[0] - 1, X_train_s.shape[1])
        pca = PCA(n_components=n_comp, random_state=config.RANDOM_STATE)
        X_train_s = pca.fit_transform(X_train_s)
        X_test_s = pca.transform(X_test_s)
        print(f"PCA reduced features to {n_comp} components")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=config.RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=3),
    }

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )

    results = []
    best_name = None
    best_f1 = -1.0
    best_clf = None

    for name, clf in models.items():
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results.append(
            {
                "model": name,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm,
            }
        )
        print(f"\n{name}: Acc={acc:.4f} F1={f1:.4f}")
        if f1 >= best_f1:
            best_f1 = f1
            best_name = name
            best_clf = clf

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, config.BEST_MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    if pca is not None:
        joblib.dump(pca, config.PCA_PATH)
    joblib.dump(le, config.LABEL_ENCODER_PATH)

    metadata = {
        "best_model": best_name,
        "best_f1": best_f1,
        "all_results": results,
        "classes": le.classes_.tolist(),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")
    print(f"Saved to {config.BEST_MODEL_PATH}")
    return metadata


if __name__ == "__main__":
    train_all_models()
