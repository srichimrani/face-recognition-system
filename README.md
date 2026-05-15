# Web-Based Visual Recognition System using ML Techniques and Flask

**University AI Project** — Real-time **ME** vs **NOT ME** face recognition using classical machine learning, Google Teachable Machine, and Flask.

---

## Project Status

| Component | Status |
|-----------|--------|
| Dataset (`dataset/ME`, `dataset/NOT_ME`) | 143 faces processed (123 ME, 20 NOT ME) |
| Training pipeline | Complete |
| Best model | **kNN (k=3)** — 93.10% test accuracy |
| Flask web app | Running successfully |
| Report & analysis | `docs/REPORT.md`, `results/tables/PERFORMANCE_RESULTS.md` |
| Teachable Machine | Run `setup_teachable_machine.bat` after TM export |

---

## Results Summary

| Model | Accuracy | F1-Score | Deployed |
|-------|----------|----------|----------|
| **kNN** | **93.10%** | **92.09%** | Yes |
| Logistic Regression | 89.66% | 89.02% | — |
| Decision Tree | 82.76% | 81.70% | — |

Full metrics: [`results/tables/PERFORMANCE_RESULTS.md`](results/tables/PERFORMANCE_RESULTS.md)

---

## Features

- OpenCV Haar Cascade face detection
- Preprocessing: crop, resize 100×100, grayscale, normalize, PCA features
- Train & compare: Logistic Regression, Decision Tree, kNN
- Google Teachable Machine (Keras) integration
- Flask UI: webcam, upload, GREEN (ME) / RED (NOT ME), confidence %
- Evaluation charts and CSV tables

---

## Quick Start

```powershell
cd C:\Users\Sri ChiMRaNi\projects\face-recognition-system
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

```
dataset/
├── ME/           # Your face
└── NOT_ME/       # Other people
```

```powershell
python scripts\import_dataset.py
python run_pipeline.py
python scripts\generate_analysis.py   # charts + tables
python app\flask_app.py
```

Open **http://127.0.0.1:5000**

Or double-click **`setup_dataset.bat`**.

---

## Project Structure

```
face-recognition-system/
├── dataset/ME, dataset/NOT_ME     # Your images
├── src/preprocessing|training|evaluation|utils
├── app/flask_app.py               # Web server
├── models/best_model.pkl          # Deployed kNN
├── results/charts/                # Generated figures
├── results/tables/                # Metrics & PERFORMANCE_RESULTS.md
├── docs/REPORT.md                 # Full academic report
├── docs/RUBRIC_CHECKLIST.md       # Submission checklist
└── run_pipeline.py                # End-to-end training
```

See [ARCHITECTURE.md](ARCHITECTURE.md) and [DATASET_GUIDE.md](DATASET_GUIDE.md).

---

## Figures (charts + Flask screenshots)

```powershell
# Save screenshots to docs\figures\incoming\flask_me.png and flask_not_me.png
python scripts\organize_figures.py
```

Report includes **7 figures** — see [docs/APPENDIX_FIGURES.md](docs/APPENDIX_FIGURES.md).

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/REPORT.md](docs/REPORT.md) | Academic report with embedded figures |
| [docs/APPENDIX_FIGURES.md](docs/APPENDIX_FIGURES.md) | Figure captions for PDF |
| [results/tables/PERFORMANCE_RESULTS.md](results/tables/PERFORMANCE_RESULTS.md) | Metrics & confusion matrices |
| [docs/RUBRIC_CHECKLIST.md](docs/RUBRIC_CHECKLIST.md) | Full-marks rubric mapping |
| [docs/VIVA_QA.md](docs/VIVA_QA.md) | Viva questions & answers |
| [docs/VIDEO_SCRIPT.md](docs/VIDEO_SCRIPT.md) | 5-minute demo script |
| [docs/PRESENTATION.md](docs/PRESENTATION.md) | Slide talking points |
| [teachable_machine/TEACHABLE_MACHINE_GUIDE.md](teachable_machine/TEACHABLE_MACHINE_GUIDE.md) | TM export steps |

---

## Testing

**Automated (optional hold-out set):**

```
dataset/test/ME/
dataset/test/NOT_ME/
python scripts\run_testing_suite.py
```

**Live:** Use webcam and upload in the Flask UI under different lighting, angles, and distances (document in report).

---

## Teachable Machine (add now)

```powershell
# 1. Train at teachablemachine.withgoogle.com (classes ME, NOT_ME)
# 2. Export TensorFlow -> Keras -> download ZIP
setup_teachable_machine.bat
python scripts\compare_ml_tm.py
python app\flask_app.py
```

Guide: [teachable_machine/TEACHABLE_MACHINE_GUIDE.md](teachable_machine/TEACHABLE_MACHINE_GUIDE.md)

---

## Deployment

**Development:** `python app/flask_app.py`

**Production (Windows example):**

```powershell
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app.flask_app:app
```

Set `DEBUG = False` in `config.py` for production.

---

## Rubric Criteria

| Code | Coverage |
|------|----------|
| **C1** Understanding | Report §§3–7, code comments, VIVA |
| **C2** Implementation | Pipeline + Flask + TM integration |
| **C3** Analysis | Metrics, charts, PERFORMANCE_RESULTS, testing |

See [docs/RUBRIC_CHECKLIST.md](docs/RUBRIC_CHECKLIST.md).

---

## Final submission

```powershell
python scripts\validate_submission.py
python scripts\create_submission_zip.py
```

Then export PDF: `export_report_pdf.bat` or open `docs\REPORT.md` in Word.

**Full checklist:** [docs/FINAL_STEPS.md](docs/FINAL_STEPS.md) | **One-page summary:** [docs/ONE_PAGE_SUMMARY.md](docs/ONE_PAGE_SUMMARY.md)

---

## Recommendations (post-submission improvements)

1. Add more **NOT_ME** images (target 50+) to improve minority-class recall.  
2. Export Teachable Machine model for ML vs TM comparison in report §10.

---

## License & Author

Academic project — replace with your name and student ID in `docs/REPORT.md` before submission.
