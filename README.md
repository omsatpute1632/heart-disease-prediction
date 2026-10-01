# 🫀 CardioPulse AI — All-in-One Python Heart Disease Prediction Web App

A **100% Python** full-stack machine learning web application for **Cardiovascular Disease Risk Prediction**, trained on the authentic **UCI Cleveland Clinic `heart.csv` dataset** and optimized for **1-click deployment on Vercel Serverless Python**.

---

## 🌟 Features

- **100% Python Stack**: Both the interactive UI, API endpoints, and Machine Learning inference are driven entirely in Python ([`api/index.py`](file:///c:/Users/omsat/heart%20predict/api/index.py) & [`app.py`](file:///c:/Users/omsat/heart%20predict/app.py)).
- **Zero Complex Static Builds**: No separate frontend build steps or complex asset routing.
- **Self-Healing Serverless Architecture**: Loads the pre-trained Scikit-Learn pipeline from `model/heart_model.joblib` with an automated fallback trainer, ensuring zero cold-start failures on Vercel.
- **Clinical Performance**:
  - **Holdout Test Accuracy**: **91.80%**
  - **ROC-AUC Score**: **95.02%**
  - **Holdout F1 Score**: **91.53%**
- **Modern Medical Dashboard**:
  - Interactive clinical parameter controls (sliders, synced inputs, pill radios)
  - 1-Click Instant Demo Presets (Healthy Adult, Borderline, High Risk)
  - SVG Radial Risk Gauge with dynamic animated color fill (Emerald &rarr; Amber &rarr; Crimson)
  - Detailed breakdown of contributing clinical risk factors and tailored medical guidance

---

## 📁 Project Structure

```text
heart-predict/
├── api/
│   └── index.py            # Complete All-in-One Python App (UI + API + ML inference)
├── data/
│   └── heart.csv           # Authentic UCI Cleveland Clinic dataset (303 patient records)
├── model/
│   ├── heart_model.joblib  # Pre-trained Scikit-Learn Random Forest pipeline
│   └── metrics.json        # Accuracy, ROC-AUC, and feature rankings
├── scripts/
│   └── train.py            # Full evaluation and model training script
├── app.py                  # Local Python runner (run: python app.py)
├── requirements.txt        # Pinned lightweight dependencies for Vercel
├── vercel.json             # Rewrites all routes directly to the Python entrypoint
├── .gitignore              # Standard git ignore patterns
└── README.md               # Documentation
```

---

## 🚀 Running Locally (1 Command)

```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Start the all-in-one Python app
python app.py
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser!

*(You can also view automatic Swagger API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs))*

---

## ☁️ Deploying to Vercel

Because everything is in Python and [`vercel.json`](file:///c:/Users/omsat/heart%20predict/vercel.json) routes all requests to `api/index.py`, deployment takes less than 2 minutes.

### Method 1: Deploy via GitHub (Recommended)

1. Push this folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "feat: All-in-one Python Heart Disease Prediction app"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
   git push -u origin main
   ```

2. Open [vercel.com](https://vercel.com) and log in.
3. Click **"Add New..."** &rarr; **"Project"** and select your GitHub repo.
4. Click **"Deploy"** (Vercel automatically detects `vercel.json` and `requirements.txt`).

### Method 2: Deploy via Vercel CLI

```bash
npm install -g vercel
vercel
```

Follow the prompts; Vercel will immediately deploy your Python application and generate your live production URL (e.g. `https://cardiopulse-ai.vercel.app`).

---

## 🔌 API Endpoints

- **`GET /`**: Returns the complete interactive web dashboard.
- **`POST /api/predict`** (or **`POST /predict`**): Evaluates patient data and returns disease probability, risk tier, contributing biomarkers, and recommendations.
- **`GET /api/health`**: Returns system health status and model availability.

---

## ⚕️ Medical Disclaimer

*This application is created for educational, triage demonstration, and screening purposes. Machine learning probability scores do not replace professional clinical judgment. Always consult a certified medical doctor or cardiologist for diagnostic evaluations and medical treatment.*
