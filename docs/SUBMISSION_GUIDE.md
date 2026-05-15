# Final Submission Guide

## 1. Organize all figures for the report

```powershell
cd C:\Users\Sri ChiMRaNi\projects\face-recognition-system
.\venv\Scripts\activate

# Charts (if not already in results/charts/)
python scripts\generate_analysis.py

# Save Flask screenshots as:
#   docs\figures\incoming\flask_me.png
#   docs\figures\incoming\flask_not_me.png

python scripts\organize_figures.py
```

Figures are embedded in `docs/REPORT.md` and listed in `docs/APPENDIX_FIGURES.md`:

| File | Content |
|------|---------|
| fig01 | Dataset distribution (123 ME, 20 NOT ME) |
| fig02 | 4-metric bar chart (LR, DT, kNN) |
| fig03 | Three confusion matrices |
| fig04 | F1-score ranking (kNN best) |
| fig05 | Best kNN confusion matrix |
| fig06 | Flask — ME, 100% confidence |
| fig07 | Flask — NOT ME, 66.67% confidence |

## 2. Export report to PDF

1. Open `docs/REPORT.md`
2. Fill title page: name, student ID, university
3. Convert to PDF (Word, Pandoc, or print-to-PDF)
4. Target: ≤15 pages main body + appendices

## 3. Submission package

| File | Required |
|------|----------|
| PDF report | Yes |
| Full project folder or ZIP | Yes |
| 5-min demo video | If required |
| Screenshots of Flask UI | Recommended |

## 4. Optional — strengthen grade

- Export Teachable Machine → `models/teachable_machine/`
- Add 30+ more NOT_ME images and re-run `python run_pipeline.py`
- Add Flask screenshots to report appendix

## 5. Verify rubric

Open `docs/RUBRIC_CHECKLIST.md` and confirm all DONE items.
