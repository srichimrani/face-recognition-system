# Google Teachable Machine — Complete Setup Guide

Use the **same images** as `dataset/ME/` and `dataset/NOT_ME/` for a fair comparison with classical ML (kNN).

---

## Part 1 — Train in Browser (15–20 minutes)

### Step 1: Open Teachable Machine

Go to: **https://teachablemachine.withgoogle.com/**

Click **Get Started** → **Image Project** → **Standard image model**

### Step 2: Create classes (names must match)

| Class name in TM | Your folder |
|------------------|-------------|
| `ME` | `dataset/ME/` |
| `NOT_ME` or `NOT ME` | `dataset/NOT_ME/` |

> Use exactly **ME** and **NOT_ME** (underscore) if possible — Flask maps both styles automatically.

### Step 3: Upload images

- Open `dataset/ME/` on your PC → select all → upload to class **ME**
- Open `dataset/NOT_ME/` → upload to class **NOT_ME**
- Aim for **all** images you used in `run_pipeline.py` (123 ME + 20 NOT ME)

### Step 4: Train

1. Click **Train Model** (wait until complete)
2. Test in the preview: your face → ME, other person → NOT_ME

### Step 5: Export

1. Click **Export Model**
2. Select **TensorFlow**
3. Choose **Keras** (`.h5` file inside ZIP)
4. Click **Download my model** (ZIP file)

---

## Part 2 — Install in This Project (2 minutes)

### Option A — Automatic (recommended)

```powershell
cd C:\Users\Sri ChiMRaNi\projects\face-recognition-system
.\venv\Scripts\activate
python scripts\setup_teachable_machine.py "C:\Users\YourName\Downloads\converted_keras.zip"
```

Or double-click **`setup_teachable_machine.bat`** and paste your ZIP path.

### Option B — Manual copy

Extract the ZIP and copy into:

```
models/teachable_machine/
├── keras_model.h5
└── labels.txt
```

Then verify:

```powershell
python scripts\setup_teachable_machine.py --verify
```

Expected output: `OK — Teachable Machine loaded`

---

## Part 3 — Compare with Classical ML

```powershell
python scripts\compare_ml_tm.py
```

Creates:

| File | Content |
|------|---------|
| `results/tables/ml_vs_tm_comparison.csv` | Side-by-side metrics |
| `results/charts/ml_vs_tm_comparison.png` | Bar chart |
| `results/tables/ML_VS_TM_RESULTS.md` | Report-ready table |

Copy results into `docs/REPORT.md` Section 9.3 and 10.2.

---

## Part 4 — Use in Flask

```powershell
python app\flask_app.py
```

Open http://127.0.0.1:5000

| UI option | Action |
|-----------|--------|
| **Google Teachable Machine** | TM only |
| **Compare Both Models** | kNN + TM side by side |

Take screenshots for **Figure 8** (optional) in your report.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| TM option disabled in UI | Run setup script; check `keras_model.h5` exists |
| `load_model` error | Re-export as Keras from TM (not TFLite only) |
| Wrong labels | Check `labels.txt` — line format: `0 ME` / `1 NOT_ME` |
| Low NOT_ME accuracy | Add more NOT_ME images in TM and retrain |
| TensorFlow slow first run | Normal — model loads once per server start |

---

## For Your Report (copy-paste points)

**Advantages:** No ML coding for training; CNN learns features; fast to prototype; built-in augmentation.

**Disadvantages:** Less control than sklearn; larger model file; needs TensorFlow at runtime; training is a black box.

**Comparison:** Run `compare_ml_tm.py` and discuss which model has higher F1 on the **same** 29-image test split.
