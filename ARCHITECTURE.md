# Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Flask Web Application                        │
│  (Webcam / Upload) → API → Classical ML  OR  Teachable Machine   │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                       ▼
┌─────────────────────┐              ┌─────────────────────┐
│  Classical ML Path  │              │ Teachable Machine   │
│  sklearn pipeline   │              │ TensorFlow/Keras    │
└──────────┬──────────┘              └──────────┬──────────┘
           │                                    │
           └────────────────┬───────────────────┘
                            ▼
                 ┌─────────────────────┐
                 │  OpenCV Face Detect │
                 │  Crop + Resize      │
                 │  Normalize Features │
                 └─────────────────────┘
```

## Folder Structure

```
face-recognition-system/
├── app/
│   └── flask_app.py           # Web server & API
├── src/
│   ├── preprocessing/         # Dataset pipeline
│   ├── training/              # ML model training
│   ├── evaluation/            # Metrics & charts
│   └── utils/                 # Face detect, features, predict
├── dataset/
│   ├── ME/                    # Your face images (primary)
│   ├── NOT_ME/                # Other people
│   └── test/ME, test/NOT_ME   # Hold-out testing
├── data/
│   └── processed/             # Cropped faces + features
├── models/
│   ├── best_model.pkl         # Best sklearn model
│   └── teachable_machine/     # TM Keras export
├── static/                    # CSS, JS
├── templates/                 # HTML
├── results/                   # Charts & tables
├── docs/                      # Report, viva, presentation
├── config.py
├── run_pipeline.py
└── requirements.txt
```

## Prediction Flow

1. User captures webcam frame or uploads image
2. Image sent as base64 to `/api/predict`
3. OpenCV detects largest face (Haar Cascade)
4. Face cropped and resized to 100×100
5. **ML path**: flatten → scale → PCA → sklearn predict
6. **TM path**: resize to model input → Keras predict
7. JSON response: label, confidence %, GREEN/RED indicator

## Training Flow

1. Raw images → `preprocess_dataset.py`
2. Features saved as `X.npy`, `y.npy`
3. `train_models.py` trains LR, DT, kNN; saves best by F1
4. `evaluate_models.py` generates confusion matrices & charts
