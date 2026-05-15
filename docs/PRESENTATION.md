# Presentation Talking Points (10–12 slides)

## Slide 1 — Title
- Project title, your name, course, university, date
- One-line tagline: "Real-time ME vs NOT ME face recognition"

## Slide 2 — Problem & Motivation
- Identity verification use cases (access control, personalization)
- Why binary classification is a good learning project
- Technologies: Python, OpenCV, sklearn, TensorFlow, Flask

## Slide 3 — Objectives
- Build dataset (ME / NOT ME)
- Preprocess with face detection
- Train & compare LR, DT, kNN
- Integrate Teachable Machine
- Deploy Flask web app with webcam

## Slide 4 — System Architecture
- Use diagram from ARCHITECTURE.md
- Data flow: Input → Face Detect → Features → Model → UI

## Slide 5 — Dataset
- Folders: `dataset/ME/`, `dataset/NOT_ME/`
- **143** processed faces: **123 ME**, **20 NOT ME**
- Ratio 6.15:1 — stratified train/test split
- Variety: lighting, angle, background, WhatsApp/Snapchat sources

## Slide 6 — Preprocessing
- Haar Cascade explanation
- Resize 100×100, grayscale, normalization
- Show before/after face crop images

## Slide 7 — Classical ML Models
- Brief theory: LR, Decision Tree, kNN
- Training: 80/20 split, scaler, PCA
- Best model selection criterion: F1-score

## Slide 8 — Teachable Machine
- Screenshot of TM training
- Export workflow
- Pros/cons table (2 columns)

## Slide 9 — Flask Application
- Screenshot of web UI
- GREEN / RED indicators
- API endpoints: `/api/predict`, `/api/predict/both`

## Slide 10 — Results
- kNN: **93.10%** acc, **92.09%** F1 (best)
- LR: 89.66% | DT: 82.76%
- Confusion matrix: 25/25 ME correct; 2/4 NOT ME errors
- Charts: `results/charts/`

## Slide 11 — Testing & Limitations
- Test matrix: lighting × angle × distance
- Challenges: no face, poor light, similar-looking people
- Ethical note: privacy, consent for photos

## Slide 12 — Conclusion & Future Work
- Achieved all rubric criteria (C1, C2, C3)
- Future: better detector, more data, cloud deploy, liveness detection
- Q&A

---

## Delivery Tips
- Spend 40% of time on live demo
- Keep theory slides visual, not text-heavy
- Prepare backup screenshots if webcam fails
