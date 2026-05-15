# Viva Voce — Questions and Answers

## AI Concepts (C1)

**Q1: What is the main objective of your project?**  
**A:** To build a web-based face recognition system that classifies faces as "ME" or "NOT ME" using conventional machine learning (Logistic Regression, Decision Tree, kNN), Google Teachable Machine, and a Flask application for real-time prediction via webcam and image upload.

**Q2: What is your dataset and how is it organized?**  
**A:** A binary image dataset with two classes. Personal images go in `data/raw/me/` and images of other people in `data/raw/not_me/`. After preprocessing, cropped faces and feature vectors are stored in `data/processed/`.

**Q3: Explain your preprocessing pipeline.**  
**A:** (1) Load raw image, (2) detect face using Haar Cascade, (3) crop largest face, (4) resize to 100×100, (5) convert to grayscale, (6) normalize pixels to 0–1, (7) flatten to feature vector, (8) optionally apply StandardScaler and PCA before training.

**Q4: Why use Haar Cascade for face detection?**  
**A:** It is fast, built into OpenCV, requires no GPU, and works well for frontal faces in real-time web applications. It is a classical computer vision approach suitable for academic demonstration.

**Q5: What features do you extract for classical ML?**  
**A:** Normalized flattened pixel intensities from the cropped grayscale face. PCA reduces dimensionality to avoid curse of dimensionality and improve generalization with small datasets.

---

## Machine Learning (C1/C3)

**Q6: Explain Logistic Regression in your project.**  
**A:** It learns a linear decision boundary in feature space to separate ME vs NOT ME. It outputs class probabilities via sigmoid/softmax, which we use for confidence percentage in the web app.

**Q7: Why compare Decision Tree and kNN?**  
**A:** Decision Tree captures non-linear rules and is interpretable. kNN classifies by similarity to nearest training faces—intuitive for face-like patterns. Comparing three algorithms demonstrates model selection based on metrics.

**Q8: What evaluation metrics did you use and why?**  
**A:** Accuracy, Precision, Recall, F1-score, and Confusion Matrix. F1 balances precision and recall, which matters when misclassifying "ME" as "NOT ME" (or vice versa) has different practical implications.

**Q9: How do you select the best model?**  
**A:** All three models are trained on the same train/test split. The model with the highest weighted F1-score on the test set is saved as `best_model.pkl` and used in Flask.

**Q10: What is overfitting and how did you reduce it?**  
**A:** Overfitting is when the model memorizes training data. We use train/test split, PCA, limited tree depth (max_depth=10), and diverse training images across lighting and angles.

---

## Teachable Machine (C1/C2)

**Q11: What is Google Teachable Machine?**  
**A:** A no-code platform to train image classifiers in the browser. We export the model as Keras `.h5` and load it in Flask with TensorFlow for comparison with sklearn models.

**Q12: Advantages of Teachable Machine?**  
**A:** Fast setup, no deep learning coding required, built-in augmentation, easy export—good for prototyping and small datasets.

**Q13: Disadvantages of Teachable Machine?**  
**A:** Less control over architecture and hyperparameters, larger model file, requires manual export step, potential overfitting with very few images, dependency on TensorFlow at runtime.

**Q14: How do classical ML and TM differ in your system?**  
**A:** Classical ML uses hand-crafted features (flattened pixels + PCA) and lightweight sklearn models. TM uses a neural network trained end-to-end on raw images (typically 224×224 RGB) with learned internal features.

---

## Flask & Implementation (C2)

**Q15: How does the Flask application work?**  
**A:** Flask serves HTML/CSS/JS. JavaScript captures webcam frames or file uploads as base64. POST requests go to `/api/predict` or `/api/predict/both`. Backend decodes image, runs face detection and selected model, returns JSON with label, confidence, and annotated image.

**Q16: How do GREEN and RED indicators work?**  
**A:** Frontend applies CSS class `me` (green) when `is_me` is true, and `not-me` (red) otherwise. A circular status indicator changes color accordingly.

**Q17: What happens if no face is detected?**  
**A:** API returns `success: false` with an error message. UI shows neutral state and prompts user to improve lighting or face position.

**Q18: What libraries did you use?**  
**A:** OpenCV (vision), scikit-learn (ML), TensorFlow (TM), Flask (web), NumPy, Matplotlib/Seaborn (evaluation charts).

---

## Testing & Analysis (C3)

**Q19: How did you test under different conditions?**  
**A:** We collected test images for varied lighting, distance, angle, background, and resolution in `data/test/`. The testing script records per-image predictions and computes accuracy for the report.

**Q20: What are the main limitations?**  
**A:** Single-face assumption (largest face only), sensitivity to extreme angles/occlusion, small dataset risk, Haar Cascade less robust than deep detectors, webcam quality affects live performance.

**Q21: What improvements would you suggest?**  
**A:** Use MTCNN or MediaPipe for detection, add data augmentation, collect more training data, implement liveness detection, deploy with HTTPS, use face embeddings (FaceNet) for better generalization.

**Q22: What is the prediction flow in one sentence?**  
**A:** Image → face detection → preprocessing → model inference → class label + probability → JSON → UI indicator.
