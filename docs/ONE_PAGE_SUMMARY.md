# One-Page Project Summary (For Examiner / Cover Sheet)

**Title:** Web-Based Visual Recognition System using ML Techniques and Flask  

**Problem:** Binary face classification — identify the authorized user (**ME**) vs everyone else (**NOT ME**).

**Dataset:** `dataset/ME/` (123 images), `dataset/NOT_ME/` (20 images) → 143 faces after OpenCV detection.

**Methods:**
- Preprocessing: Haar Cascade → crop 100×100 → grayscale → normalize → PCA (50)
- Classifiers: Logistic Regression, Decision Tree, kNN (k=3)
- **Best model:** kNN — **93.10% accuracy**, **92.09% F1** (test n=29)
- Web: Flask + webcam + upload; GREEN/RED UI; confidence %
- Teachable Machine: integration ready (`models/teachable_machine/`)

**Key results (kNN confusion matrix):**
- ME: 25/25 correct (100% recall)
- NOT ME: 2/4 correct (50% recall); class imbalance noted

**Live demo:** Flask verified — ME at 100% confidence; NOT ME at 66.67% (see Figures 6–7).

**Deliverables:** Source code, `docs/REPORT.md` (PDF), 7 figures, VIVA Q&A, demo video script.

**Student:** [Your Name] | **ID:** [Your ID] | **Date:** May 2026
