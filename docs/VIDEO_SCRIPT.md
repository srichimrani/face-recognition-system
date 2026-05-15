# 5-Minute Video Demonstration Script

**Total time: ~5 minutes** | Speak clearly, show screen + face to camera

---

## [0:00 – 0:30] Introduction

> "Hello. I am [Your Name], and this is my Artificial Intelligence project titled **Web-Based Visual Recognition System using ML Techniques and Flask**.
>
> The system identifies whether a face is **ME** or **NOT ME** using classical machine learning, Google Teachable Machine, and a Flask web application with real-time webcam recognition."

*[Show title slide or project folder in IDE]*

---

## [0:30 – 1:15] Dataset & Preprocessing

> "I collected a personal dataset: my images in the ME class, and other people's images in the NOT ME class.
>
> The preprocessing pipeline uses OpenCV Haar Cascade for face detection, crops the face, resizes to 100 by 100, converts to grayscale, and normalizes pixel values for feature extraction."

*[Open `data/raw/me` and `data/processed` folders; show 2–3 sample cropped faces]*

Run briefly in terminal (optional):
```bash
python run_pipeline.py
```
> "This trains Logistic Regression, Decision Tree, and kNN, and selects the best model by F1-score."

---

## [1:15 – 2:00] Training Results

> "Here are the evaluation metrics and confusion matrices generated automatically."

*[Open `results/charts/ml_metrics_comparison.png` and `confusion_matrices.png`]*
> "k-Nearest Neighbors achieved the best results: 93.10% accuracy and 92.09% F1-score on the test set, with zero errors classifying me as NOT ME."

---

## [2:00 – 2:45] Teachable Machine

> "I also trained an image model on Google Teachable Machine with the same two classes, exported it as Keras, and placed it in the models folder."

*[Show Teachable Machine website or exported files in `models/teachable_machine/`]*

> "Teachable Machine uses a neural network that learns features automatically, while classical ML uses extracted pixel features with PCA."

---

## [2:45 – 4:15] Live Web Demo (MAIN SECTION)

Start Flask:
```bash
python app/flask_app.py
```

Open **http://127.0.0.1:5000**

> "This is the Flask web interface. I will demonstrate webcam recognition."

**Demo 1 — ME (your face):**
- Click **Start Webcam** → **Capture & Predict**
> "GREEN indicator — identified as ME with [X]% confidence."

**Demo 2 — NOT ME (friend/photo on phone):**
> "RED indicator — NOT ME."

**Demo 3 — Upload image:**
- Switch to **Upload Image**, select a test photo
> "Works with uploaded images as well."

**Demo 4 — Compare both models:**
- Select **Compare Both Models**
> "Here we see classical ML and Teachable Machine side by side."

---

## [4:15 – 4:45] Testing Discussion

> "I tested under different lighting, distances, angles, and backgrounds using the Flask app. The deployed kNN model reached 93% on the held-out test set. Two NOT ME faces were misclassified as ME due to class imbalance—we have 123 ME versus 20 NOT ME images. Teachable Machine comparison is ready once I export the Keras model."

*[Optional: show `results/tables/live_testing_results.csv`]*

---

## [4:45 – 5:00] Conclusion

> "In conclusion, this project demonstrates the full AI pipeline: dataset, preprocessing, multiple ML algorithms, Teachable Machine integration, evaluation, and a Flask deployment for real-time face recognition. Thank you."

*[End screen: project GitHub/path + your name]*

---

## Recording Checklist

- [ ] Good lighting on your face
- [ ] Webcam permission allowed in browser
- [ ] Models trained before recording
- [ ] TM model exported (if showing TM demo)
- [ ] Screen recorder captures both IDE and browser
- [ ] Audio clear, no background noise
