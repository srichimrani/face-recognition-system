# Performance Results — Experimental Run (Completed)

**Date:** May 2026 | **Pipeline:** `run_pipeline.py` | **Deployment:** Flask verified

---

## 1. Dataset Summary (After Preprocessing)

| Metric | Value |
|--------|-------|
| Total face images processed | **143** |
| Class ME | **123** (86.0%) |
| Class NOT ME | **20** (14.0%) |
| Source folders | `dataset/ME/`, `dataset/NOT_ME/` |
| Face crop size | 100 × 100 pixels (grayscale) |
| Raw feature dimension | 10,000 (flattened pixels) |
| PCA components (training) | 50 |

**Note:** Class imbalance (more ME than NOT ME samples) is discussed in the report; stratified splitting was used to preserve class proportions in train/test sets.

---

## 2. Train / Test Configuration

| Setting | Value |
|---------|-------|
| Split ratio | 80% train / 20% test |
| Train samples | 114 |
| Test samples | 29 |
| Stratification | Yes (`stratify=y`) |
| Random seed | 42 |
| Preprocessing | StandardScaler + PCA(50) |

---

## 3. Classical ML Model Comparison (Held-Out Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | Selected |
|-------|----------|-----------|--------|----------|----------|
| **k-Nearest Neighbors (k=3)** | **93.10%** | **93.61%** | **93.10%** | **92.09%** | **Best — deployed** |
| Logistic Regression | 89.66% | 88.77% | 89.66% | 89.02% | — |
| Decision Tree (max_depth=10) | 82.76% | 80.86% | 82.76% | 81.70% | — |

**Selection criterion:** Highest weighted **F1-score** on the test set.

---

## 4. Best Model (kNN) — Confusion Matrix

Class order: `[ME, NOT ME]` (LabelEncoder alphabetical)

|  | Predicted ME | Predicted NOT ME |
|--|:------------:|:--------------:|
| **Actual ME** | **25** (TP) | **0** (FN) |
| **Actual NOT ME** | **2** (FP) | **2** (TN) |

### Interpretation

| Class | Precision | Recall | Comment |
|-------|-----------|--------|---------|
| ME | 25/27 = 92.6% | 25/25 = **100%** | Excellent recognition of authorized user |
| NOT ME | 2/2 = 100% | 2/4 = **50%** | Two non-user faces misclassified as ME |

Overall test accuracy **93.10%** with **3 errors** out of 29 test images (all errors involve NOT ME samples).

---

## 5. Logistic Regression — Confusion Matrix

|  | Pred ME | Pred NOT ME |
|--|---------|-------------|
| Actual ME | 24 | 1 |
| Actual NOT ME | 2 | 2 |

---

## 6. Decision Tree — Confusion Matrix

|  | Pred ME | Pred NOT ME |
|--|---------|-------------|
| Actual ME | 23 | 2 |
| Actual NOT ME | 3 | 1 |

---

## 7. Google Teachable Machine

| Item | Status |
|------|--------|
| Flask integration | Implemented (`TeachableMachinePredictor`) |
| Model file | Pending — place `keras_model.h5` + `labels.txt` in `models/teachable_machine/` |
| Comparison | Use web UI **Compare Both Models** after export |

See `teachable_machine/TEACHABLE_MACHINE_GUIDE.md`.

---

## 8. Live System Testing (Flask Web Application)

| Test ID | Condition | Result | Confidence | Figure |
|---------|-----------|--------|------------|--------|
| L1 | Author face (portrait) | GREEN / **ME** | **100%** | Fig. 6 |
| L2 | Other person (casual) | RED / **NOT ME** | **66.67%** | Fig. 7 |
| L3 | Face bounding box | Detected | — | Figs. 6–7 |
| L4 | Model label in UI | kNN (best model) | — | Figs. 6–7 |
| L5 | Image upload | Functional | — | — |

Screenshots saved as `docs/figures/fig06_flask_demo_me.png` and `fig07_flask_demo_not_me.png`.

---

## 9. Charts (generate with)

```bash
python scripts/generate_analysis.py
```

Output files:

- `results/charts/ml_metrics_comparison.png`
- `results/charts/confusion_matrices.png`
- `results/charts/dataset_class_distribution.png`
- `results/charts/f1_score_ranking.png`
- `results/charts/best_model_confusion_matrix.png`

---

## 10. Recommendations from Results

1. Add **more NOT_ME images** (target 50+）to improve minority-class recall.
2. Export **Teachable Machine** model for full ML vs TM comparison table.
3. Use `dataset/test/ME` and `dataset/test/NOT_ME` for automated metrics: `python scripts/run_testing_suite.py`
