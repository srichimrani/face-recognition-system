# Appendix — Figures for Project Report

Insert these figures into `docs/REPORT.md` or your PDF export.  
Run `python scripts/organize_figures.py` after saving screenshots to `docs/figures/incoming/`.

---

## Figure 1 — Dataset Class Distribution

![Dataset class distribution](../figures/fig01_dataset_class_distribution.png)

**Figure 1.** Number of face images per class after preprocessing (ME: 123, NOT ME: 20). The imbalance ratio is approximately 6.15:1.

---

## Figure 2 — Classical ML Metrics Comparison

![ML metrics comparison](../figures/fig02_ml_metrics_comparison.png)

**Figure 2.** Comparison of Accuracy, Precision, Recall, and F1-Score for Logistic Regression, Decision Tree, and kNN on the held-out test set (n = 29).

---

## Figure 3 — Confusion Matrices (All Models)

![Confusion matrices all models](../figures/fig03_confusion_matrices_all_models.png)

**Figure 3.** Confusion matrices for Logistic Regression, Decision Tree, and kNN. Test set contains 25 ME and 4 NOT ME samples (stratified split).

---

## Figure 4 — F1-Score Ranking (Model Selection)

![F1 score ranking](../figures/fig04_f1_score_ranking.png)

**Figure 4.** Weighted F1-score used to select the best model. **kNN (0.921)** was chosen for deployment.

---

## Figure 5 — Best Model (kNN) Confusion Matrix

![Best model confusion matrix](../figures/fig05_best_model_knn_confusion_matrix.png)

**Figure 5.** Confusion matrix for deployed kNN model: 25/25 ME correct; 2/4 NOT ME correct; 2 false positives (NOT ME classified as ME).

---

## Figure 6 — Flask Demo: ME (Authorized User)

![Flask demo ME](../figures/fig06_flask_demo_me.png)

**Figure 6.** Live Flask application — webcam/upload prediction. **GREEN** indicator, label **ME**, confidence **100%**, model **kNN**, face bounding box shown.

---

## Figure 7 — Flask Demo: NOT ME

![Flask demo NOT ME](../figures/fig07_flask_demo_not_me.png)

**Figure 7.** Live Flask application — different subject classified as **NOT ME** with **RED** indicator, confidence **66.67%**, model **kNN**.

---

## Figure 8 — ML vs Teachable Machine (after `compare_ml_tm.py`)

![ML vs TM comparison](../figures/fig08_ml_vs_tm_comparison.png)

**Figure 8.** Classical ML (kNN) vs Teachable Machine on the same test split.

---

## Quick copy checklist

| Save as | Source |
|---------|--------|
| `incoming/flask_me.png` | Your ME screenshot |
| `incoming/flask_not_me.png` | Your NOT ME screenshot |
| Charts | Auto from `results/charts/` via `organize_figures.py` |
