# Dataset Guide

## Folder Structure (Required)

Place your images in the project under:

```
face-recognition-system/
└── dataset/
    ├── ME/           ← Your face photos
    ├── NOT_ME/       ← Other people's photos
    └── test/         ← Optional hold-out set for report testing
        ├── ME/
        └── NOT_ME/
```

## Requirements

| Class | Folder | Minimum (recommended) |
|-------|--------|------------------------|
| You | `dataset/ME/` | 15–30+ images |
| Others | `dataset/NOT_ME/` | 15–30+ images |

**Formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

Images in subfolders are supported (recursive scan).

## Quick Commands

```powershell
cd C:\Users\Sri ChiMRaNi\projects\face-recognition-system

# Verify dataset counts
python scripts/import_dataset.py

# Preprocess + train + evaluate
python run_pipeline.py

# Optional: mirror copy to data/raw/
python scripts/import_dataset.py --copy
```

## After Training

- Cropped faces: `data/processed/ME/`, `data/processed/NOT_ME/`
- Charts: `results/charts/`
- Best model: `models/best_model.pkl`

## Teachable Machine

Use the **same** `ME` and `NOT_ME` images when training on [Teachable Machine](https://teachablemachine.withgoogle.com/).
