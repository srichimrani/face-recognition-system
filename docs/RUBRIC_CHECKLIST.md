# Rubric Checklist — Full Marks Verification

Use this checklist before submission. All items marked **DONE** based on completed training and Flask deployment.

---

## C1 — Understanding of AI Concepts

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Dataset explained | REPORT §3, `DATASET_GUIDE.md`, 143 samples documented | DONE |
| ML algorithms explained | REPORT §5 (LR, DT, kNN theory + use) | DONE |
| Working principles | REPORT §4 preprocessing, §5 prediction flow | DONE |
| Training process | `run_pipeline.py`, REPORT §5.3 | DONE |
| Prediction flow | REPORT §5.4, §7, ARCHITECTURE.md | DONE |
| Simple professional language | Full report + VIVA_QA.md | DONE |

---

## C2 — Implementation

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Conventional ML | 3 models trained; kNN deployed | DONE |
| Teachable Machine | `setup_teachable_machine.bat`, `compare_ml_tm.py` | INSTALL* |
| Flask integration | `app/flask_app.py` — verified running | DONE |
| Webcam input | `static/js/app.js` MediaDevices API | DONE |
| Image upload | File input + base64 API | DONE |
| GREEN / RED indicators | CSS `.me` / `.not-me` panels | DONE |
| Confidence % | API `confidence` field in UI | DONE |
| No broken code | Pipeline + Flask completed successfully | DONE |
| Professional structure | `src/`, `app/`, `dataset/`, `docs/` | DONE |

\*Run `setup_teachable_machine.bat` then `compare_ml_tm.py` after TM export.

---

## C3 — Analysis

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Accuracy | 93.10% (kNN test set) | DONE |
| Precision | 93.61% weighted | DONE |
| Recall | 93.10% weighted | DONE |
| F1-score | 92.09% — model selection metric | DONE |
| Confusion matrix | REPORT §9.2 + metadata JSON | DONE |
| Comparison of models | Table §9.1, PERFORMANCE_RESULTS.md | DONE |
| Charts + Flask screenshots | `docs/figures/` fig01–fig07, REPORT §8–9 | DONE |
| Testing | Live Flask + test script documented | DONE |
| Strengths / weaknesses | REPORT §10.3–10.4 | DONE |
| Limitations | REPORT §12 | DONE |
| Recommendations | REPORT §14, PERFORMANCE_RESULTS §10 | DONE |

\*\*Run once: `python scripts/generate_analysis.py` to create PNG charts for report appendix.

---

## Deliverables Checklist

| Item | File / Location |
|------|-----------------|
| Source code | `src/`, `app/`, `config.py` |
| Report | `docs/REPORT.md` → export PDF |
| README | `README.md` |
| requirements.txt | Root |
| Results tables | `results/tables/` |
| Viva prep | `docs/VIVA_QA.md` |
| Video script | `docs/VIDEO_SCRIPT.md` |
| Presentation | `docs/PRESENTATION.md` |

---

## Pre-Submission Actions

- [ ] Fill name, ID, university on REPORT title page
- [x] Charts generated (fig01–fig05)
- [x] Flask screenshots (fig06 ME 100%, fig07 NOT ME 66.67%)
- [x] Figures organized (`docs/figures/`)
- [ ] Run `python scripts\validate_submission.py`
- [ ] Export `docs\REPORT.md` → PDF (`export_report_pdf.bat`)
- [ ] Run `python scripts\create_submission_zip.py`
- [ ] Record 5-min demo (`docs\VIDEO_SCRIPT.md`)
- [ ] Export Teachable Machine model (optional — strengthens C2/C3)

**See `docs/FINAL_STEPS.md` for the complete closing checklist.**

**Estimated grade alignment:** All core criteria met for full marks on C1–C3.
