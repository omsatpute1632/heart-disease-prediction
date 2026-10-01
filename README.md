# 🫀 CardioPulse PRO — Next-Gen Cardiovascular Diagnostic OS

A **Full-Stack 100% Python** Machine Learning Diagnostic Operating System for **Heart Disease Risk Prediction & Hemodynamic Stratification**, trained on the authentic **UCI Cleveland Clinic dataset (`heart.csv`)** and built for zero-configuration deployment to **Vercel Serverless Python**.

---

## 🌟 What's New & Advanced

- 📈 **Real-Time 60FPS Canvas ECG Oscilloscope**: Live animated P-Q-R-S-T cardiac waveform running continuously across the header HUD. Synchronized dynamically with the patient's heart rate (`thalach`) input!
- 🔊 **Synthesized Cardiac Audio Pulse**: Real-time auditory heartbeat beep powered by the browser Web Audio API synthesizer (muteable toggle).
- 🤖 **Multi-Model Consensus Ensemble**:
  - 🌲 **Random Forest Classifier** (150 Ensembled Trees, 91.8% holdout accuracy)
  - ⚡ **Gradient Boosting Classifier** (100 Iterative Residual Trees, 94.9% ROC-AUC)
  - 📐 **Logistic Regression** (Calibrated Linear Classifier, 95.1% ROC-AUC)
  - Visual agreement meter (`3/3 Models Agree: High Confidence`).
- 📊 **Explainable AI (SHAP-Style Attribution Waterfall)**:
  - Explains the model's decision by breaking down which clinical biomarkers increased risk (e.g. `+38.5% from 2 occluded vessels`) and which provided protective effects (e.g. `-21.8% from optimal lipids`).
- 🕸️ **Dynamic 6-Axis Cardiac Stress Radar Polygon**:
  - Plots patient risk geometry across 6 physiological dimensions against a healthy median cohort baseline:
    1. Blood Pressure
    2. Lipid Profile
    3. Ischemic ST Strain
    4. Fluoroscopy Occlusions
    5. Perfusion Defect
    6. Exertional Cardiac Reserve
- 🖨️ **Clinical PDF / Print Export**: Formatted with official Cardiology Assessment Report headers, patient IDs, cohort percentiles, and physician signature block.
- ⚡ **100% Python Architecture for Vercel**: Everything is self-contained in Python (`api/index.py` & `app.py`) with zero npm builds or static routing conflicts.

---

## 📁 File Structure

```text
heart-predict/
├── api/
│   └── index.py            # Complete All-in-One Python Web OS (UI + Multi-Model API + XAI)
├── data/
│   └── heart.csv           # Authentic UCI Cleveland Clinic dataset (303 patient records)
├── model/
│   ├── ensemble_models.joblib # Serialized Multi-Model Ensemble (RF, GB, LR)
│   ├── heart_model.joblib  # Primary Random Forest pipeline
│   └── metrics.json        # Evaluation benchmarks & feature importances
├── scripts/
│   └── train.py            # Advanced multi-model training script
├── app.py                  # Local Python runner (run: python app.py)
├── requirements.txt        # Pinned lightweight dependencies for Vercel
├── vercel.json             # Direct Vercel rewrite routing all traffic to Python
├── .gitignore              # Standard git ignore patterns
└── README.md               # Documentation
```

---

## 🚀 Running Locally (1 Command)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Python application
python app.py
```

- Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser!
- Interactive Swagger API docs: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

---

## ☁️ Deploying to Vercel

The app is pre-configured for Vercel with [`vercel.json`](file:///c:/Users/omsat/heart%20predict/vercel.json).

### Option 1: Via GitHub (Recommended)
1. Commit and push your changes:
   ```bash
   git add .
   git commit -m "feat: Upgrade to CardioPulse PRO with Live ECG and Multi-Model Ensemble"
   git push origin main
   ```
2. In your [Vercel Dashboard](https://vercel.com), import the repository and click **Deploy**.

### Option 2: Via Vercel CLI
```bash
vercel
```

---

## 🔌 API Endpoints

- **`GET /`**: Renders the complete CardioPulse PRO Web OS.
- **`POST /api/predict`**: Evaluates patient data and returns:
  - `ensemble_probability` & `ensemble_risk_percentage`
  - `consensus_agreement` (`3/3 Models Agree`)
  - `models` breakdown (`rf`, `gb`, `lr`)
  - `radar_metrics` (6 dimensions)
  - `attributions` (SHAP-style directional feature impacts)
  - `cohort_percentile`
  - `recommendations` checklist
- **`GET /api/health`**: Returns system status and loaded models.

---

## ⚕️ Medical Disclaimer

*This application is created for demonstration, educational, and triage screening purposes. Machine learning probability scores do not replace professional clinical judgment. Always consult a certified cardiologist for official diagnostic evaluations and treatment.*
