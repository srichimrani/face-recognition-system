# Final Steps — You Are Almost Done

You completed: dataset, training, Flask, charts, and screenshots. Follow this list to submit.

---

## Step 1 — Validate (2 minutes)

```powershell
cd C:\Users\Sri ChiMRaNi\projects\face-recognition-system
python scripts\validate_submission.py
```

Fix any **ERRORS**. Warnings are OK if noted in report (e.g. TM not exported).

---

## Step 2 — Export PDF report (10 minutes)

1. Open `docs\REPORT.md` in **Microsoft Word**
2. Confirm all **7 figures** display under sections 8–9
3. Edit title page: your **name**, **student ID**, **university**
4. **Save As → PDF** → `docs\REPORT.pdf`

Or double-click `export_report_pdf.bat`

---

## Step 3 — Teachable Machine (30 minutes) — **DO THIS**

Full guide: `teachable_machine\TEACHABLE_MACHINE_GUIDE.md`

```powershell
# After downloading Keras ZIP from Teachable Machine:
setup_teachable_machine.bat
python scripts\compare_ml_tm.py
python app\flask_app.py
```

- Select **Compare Both Models** in the UI
- Add `results/charts/ml_vs_tm_comparison.png` to report §9.3
- Copy table from `results/tables/ML_VS_TM_RESULTS.md` into REPORT.md

---

## Step 4 — Demo video (if required)

Follow `docs\VIDEO_SCRIPT.md` (5 minutes). Show:
- Dataset folders
- Charts (fig01–fig05)
- Flask ME (green) and NOT ME (red)

---

## Step 5 — Create submission ZIP

```powershell
python scripts\create_submission_zip.py
```

Submit:
- `face_recognition_submission_YYYYMMDD.zip`
- `docs\REPORT.pdf`
- Demo video link/file (if required)

---

## Step 6 — Viva preparation

- Read `docs\VIVA_QA.md`
- Memorize: **kNN best**, **93.1% accuracy**, **class imbalance** 123 vs 20
- Know preprocessing pipeline in 4 sentences

---

## Rubric — Final status

| Criterion | Status |
|-----------|--------|
| C1 Understanding | Report + VIVA complete |
| C2 Implementation | Pipeline + Flask working |
| C3 Analysis | Metrics + 7 figures + live tests |

**You are submission-ready after PDF export and ZIP creation.**
