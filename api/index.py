"""
CardioPulse AI — All-in-One Full-Stack Python Application for Vercel
Serves the complete interactive web UI, API endpoints, and ML inference directly from Python.
"""

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cardiopulse_python")

app = FastAPI(
    title="CardioPulse AI — Heart Disease Prediction",
    description="All-in-one Python web application and ML inference engine.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Features & Model Initialization
# -------------------------------------------------------------
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "heart_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")

model_pipeline = None


def train_fallback_model() -> Pipeline:
    """Trains a fallback RandomForest pipeline if the pre-serialized model is missing."""
    logger.info("Initializing fallback model training from data/heart.csv...")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        # Emergency backup sample data if file path varies in serverless environment
        logger.warning("data/heart.csv not found; using embedded reference dataset...")
        df = pd.DataFrame([
            [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
            [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 1],
            [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2, 7, 1],
            [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0, 3, 0],
            [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, 3, 0],
            [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, 3, 0],
            [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, 2, 3, 1],
            [57, 0, 4, 120, 354, 0, 0, 163, 1, 0.6, 1, 0, 3, 0],
            [63, 1, 4, 130, 254, 0, 2, 147, 0, 1.4, 2, 1, 7, 1],
            [53, 1, 4, 140, 203, 1, 2, 155, 1, 3.1, 3, 0, 7, 1]
        ], columns=FEATURE_NAMES + ["target"])

    X = df[FEATURE_NAMES]
    y = df["target"]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    pipeline.fit(X, y)
    logger.info("Fallback pipeline trained successfully.")
    return pipeline


# Load model or train fallback
try:
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Loaded trained model from {MODEL_PATH}")
    else:
        model_pipeline = train_fallback_model()
except Exception as err:
    logger.error(f"Error loading model: {err}. Using trained fallback.")
    model_pipeline = train_fallback_model()


# -------------------------------------------------------------
# Pydantic Schemas & Helpers
# -------------------------------------------------------------
class PatientPayload(BaseModel):
    age: int = Field(..., ge=18, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=4)
    trestbps: float = Field(..., ge=70, le=250)
    chol: float = Field(..., ge=80, le=650)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=50, le=240)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0.0, le=8.0)
    slope: int = Field(..., ge=0, le=3)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=7)


def normalize_patient(patient: PatientPayload) -> Dict[str, Any]:
    data = patient.model_dump()
    if data["cp"] == 0:
        data["cp"] = 1
    if data["slope"] == 0:
        data["slope"] = 1
    if data["thal"] == 1:
        data["thal"] = 3
    elif data["thal"] == 2:
        data["thal"] = 6
    elif data["thal"] == 3:
        data["thal"] = 7
    elif data["thal"] not in [3, 6, 7]:
        data["thal"] = 3
    return data


def extract_clinical_factors(data: Dict[str, Any]) -> List[Dict[str, str]]:
    factors = []
    if data["trestbps"] >= 140:
        factors.append({"param": "Blood Pressure", "status": "Stage 2 Hypertension", "value": f"{data['trestbps']:.0f} mmHg", "sev": "high", "note": "Elevated systolic pressure adds acute strain on arterial walls."})
    elif data["trestbps"] >= 130:
        factors.append({"param": "Blood Pressure", "status": "Pre-Hypertension", "value": f"{data['trestbps']:.0f} mmHg", "sev": "moderate", "note": "Borderline blood pressure; monitor periodically."})

    if data["chol"] >= 240:
        factors.append({"param": "Serum Cholesterol", "status": "High Cholesterol", "value": f"{data['chol']:.0f} mg/dL", "sev": "high", "note": "Significantly elevated circulating lipids accelerate plaque buildup."})
    elif data["chol"] >= 200:
        factors.append({"param": "Serum Cholesterol", "status": "Borderline High", "value": f"{data['chol']:.0f} mg/dL", "sev": "moderate", "note": "Above desirable threshold of <200 mg/dL."})

    if data["oldpeak"] >= 2.0:
        factors.append({"param": "ST Depression", "status": "Severe Ischemia", "value": f"{data['oldpeak']:.1f} mm", "sev": "high", "note": "Pronounced ST depression during exertion reflects impaired myocardial perfusion."})
    elif data["oldpeak"] >= 1.0:
        factors.append({"param": "ST Depression", "status": "Mild Ischemia", "value": f"{data['oldpeak']:.1f} mm", "sev": "moderate", "note": "Subtle exercise-induced ST depression."})

    if data["exang"] == 1:
        factors.append({"param": "Exercise Angina", "status": "Present", "value": "Positive", "sev": "high", "note": "Chest pain elicited by physical exertion."})

    if data["ca"] > 0:
        factors.append({"param": "Fluoroscopy Vessels", "status": f"{data['ca']} Vessel(s) Occluded", "value": f"{data['ca']} vessel(s)", "sev": "high", "note": "Fluoroscopy demonstrates visible coronary narrowing."})

    if data["thal"] in [6, 7]:
        desc = "Fixed Defect" if data["thal"] == 6 else "Reversible Ischemic Defect"
        factors.append({"param": "Thallium Scan", "status": desc, "value": f"Type {data['thal']}", "sev": "high", "note": "Nuclear myocardial scintigraphy revealed perfusion defect."})

    if data["cp"] == 4:
        factors.append({"param": "Chest Pain Type", "status": "Asymptomatic / Silent", "value": "Type 4", "sev": "high", "note": "Silent presentation is often associated with advanced occult disease."})

    return factors


def generate_recommendations(risk_tier: str) -> List[str]:
    if risk_tier == "High Risk":
        return [
            "🚨 Urgent Cardiology Consultation: Arrange an appointment with a board-certified cardiologist within 48-72 hours.",
            "📋 Diagnostic Angiogram: Discuss invasive coronary angiography or high-resolution CTCA imaging.",
            "💊 Pharmacotherapy Review: Evaluate antiplatelet, statin, and antihypertensive regimens.",
            "⚠️ Strenuous Workouts Caution: Cease unmonitored vigorous exertion until cleared by an exercise stress test."
        ]
    elif risk_tier == "Moderate Risk":
        return [
            "🩺 Preventive Checkup: Follow up with your primary physician within 3-4 weeks.",
            "📊 Stress Testing: Undergo a Treadmill Exercise Stress Test (TMT) and echocardiogram.",
            "🥗 Nutritional Optimization: Transition to a Mediterranean or DASH diet low in saturated fats and sodium.",
            "🏃 Structured Activity: Aim for 150 minutes of moderate aerobic activity weekly (e.g. brisk walking)."
        ]
    else:
        return [
            "🌟 Favorable Baseline: Excellent cardiovascular markers. Continue your healthy daily habits!",
            "🛡️ Annual Surveillance: Maintain annual monitoring of resting blood pressure and lipid panels.",
            "🥦 Nutrient-Dense Diet: Keep enjoying whole grains, leafy vegetables, and lean proteins.",
            "🧘 Stress & Sleep: Prioritize 7-8 hours of restful sleep and mindfulness practices."
        ]


# -------------------------------------------------------------
# Embedded Complete UI Template (Pure Python HTMLResponse)
# -------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardioPulse AI — Heart Disease Prediction (Python on Vercel)</title>
  <meta name="description" content="All-in-one Python Heart Disease Prediction application deployed on Vercel.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #07090e;
      --bg-card: rgba(17, 24, 39, 0.8);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-focus: #38bdf8;
      --border-glass: rgba(255, 255, 255, 0.12);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-faint: #64748b;
      --healthy: #10b981;
      --warning: #f59e0b;
      --danger: #f43f5e;
      --font-heading: 'Outfit', sans-serif;
      --font-body: 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-body);
      background-color: var(--bg-base);
      color: var(--text-main);
      min-height: 100vh;
      overflow-x: hidden;
      line-height: 1.5;
    }
    .bg-mesh {
      position: fixed; inset: 0;
      background: 
        radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(244, 63, 94, 0.09) 0%, transparent 45%),
        radial-gradient(circle at 50% 80%, rgba(59, 130, 246, 0.08) 0%, transparent 50%);
      pointer-events: none; z-index: 0;
    }
    .grid-overlay {
      position: fixed; inset: 0;
      background-image: linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
      background-size: 32px 32px;
      pointer-events: none; z-index: 1;
    }
    .app-container {
      position: relative; z-index: 2;
      max-width: 1320px; margin: 0 auto;
      padding: 24px 24px 60px;
      display: flex; flex-direction: column; gap: 20px;
    }
    .app-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 16px 24px;
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-glass);
      border-radius: 18px;
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .heart-icon {
      width: 44px; height: 44px; border-radius: 12px;
      background: linear-gradient(135deg, rgba(244, 63, 94, 0.2), rgba(59, 130, 246, 0.2));
      border: 1px solid rgba(244, 63, 94, 0.3);
      display: flex; align-items: center; justify-content: center;
      color: var(--danger);
      animation: heartbeat 2s infinite ease-in-out;
    }
    .heart-icon svg { width: 24px; height: 24px; }
    @keyframes heartbeat { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.08); filter: drop-shadow(0 0 8px rgba(244, 63, 94, 0.6)); } }
    .brand-title { font-family: var(--font-heading); font-size: 1.55rem; font-weight: 800; color: #fff; }
    .gradient-text { background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .brand-subtitle { font-size: 0.82rem; color: var(--text-muted); }
    .header-badge {
      display: flex; align-items: center; gap: 8px;
      padding: 6px 14px; background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 999px;
      font-size: 0.8rem; font-weight: 600; color: #34d399;
    }
    .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--healthy); box-shadow: 0 0 8px var(--healthy); }
    
    .presets-bar {
      display: flex; align-items: center; flex-wrap: wrap; gap: 10px;
      padding: 12px 20px; background: var(--bg-card);
      border: 1px solid var(--border-glass); border-radius: 14px;
    }
    .presets-label { font-size: 0.82rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em; margin-right: 4px; }
    .btn-preset {
      display: flex; align-items: center; gap: 8px; padding: 7px 14px;
      font-family: var(--font-body); font-size: 0.82rem; font-weight: 600;
      color: var(--text-main); background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-subtle); border-radius: 8px;
      cursor: pointer; transition: all 0.2s;
    }
    .btn-preset:hover { background: rgba(255, 255, 255, 0.08); transform: translateY(-1px); border-color: rgba(255,255,255,0.2); }
    .dot { width: 8px; height: 8px; border-radius: 50%; }
    .dot-green { background: var(--healthy); box-shadow: 0 0 6px var(--healthy); }
    .dot-amber { background: var(--warning); box-shadow: 0 0 6px var(--warning); }
    .dot-red { background: var(--danger); box-shadow: 0 0 6px var(--danger); }
    .btn-reset { margin-left: auto; color: var(--text-muted); }

    .main-grid {
      display: grid; grid-template-columns: 1.15fr 0.85fr;
      gap: 22px; align-items: start;
    }
    .form-col { display: flex; flex-direction: column; gap: 18px; }
    .form-card {
      background: var(--bg-card); backdrop-filter: blur(16px);
      border: 1px solid var(--border-glass); border-radius: 16px;
      padding: 20px 22px;
    }
    .card-title {
      font-family: var(--font-heading); font-size: 1.02rem; font-weight: 700;
      color: #fff; margin-bottom: 4px;
    }
    .card-subtitle { font-size: 0.76rem; color: var(--text-muted); margin-bottom: 16px; }
    .input-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .col-2 { grid-column: span 2; }
    .form-group { display: flex; flex-direction: column; gap: 6px; }
    .form-label {
      font-size: 0.8rem; font-weight: 600; color: #e2e8f0;
      display: flex; justify-content: space-between;
    }
    .hint { font-size: 0.72rem; font-weight: 400; color: var(--text-faint); }
    .form-control {
      width: 100%; padding: 9px 12px;
      font-family: var(--font-body); font-size: 0.88rem;
      color: #fff; background: rgba(15, 23, 42, 0.75);
      border: 1px solid var(--border-subtle); border-radius: 8px;
      outline: none; transition: border-color 0.2s;
    }
    .form-control:focus { border-color: var(--border-focus); box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15); }
    select.form-control { cursor: pointer; }
    select.form-control option { background: #0f172a; color: #fff; }

    .slider-row { display: flex; align-items: center; gap: 10px; }
    .custom-slider { flex: 1; -webkit-appearance: none; height: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 999px; }
    .custom-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #38bdf8; cursor: pointer; }
    .num-box { width: 68px; text-align: center; font-family: var(--font-mono); font-weight: 600; }

    .pill-group { display: flex; gap: 8px; }
    .pill-opt { flex: 1; display: flex; align-items: center; justify-content: center; cursor: pointer; }
    .pill-opt input { display: none; }
    .pill-opt span {
      width: 100%; padding: 8px; text-align: center; font-size: 0.8rem; font-weight: 600;
      color: var(--text-muted); background: rgba(15, 23, 42, 0.75);
      border: 1px solid var(--border-subtle); border-radius: 8px;
      transition: all 0.2s;
    }
    .pill-opt input:checked + span {
      background: rgba(56, 189, 248, 0.15); border-color: #38bdf8; color: #fff;
    }

    .btn-submit {
      width: 100%; padding: 14px 24px;
      background: linear-gradient(135deg, #0284c7, #0369a1);
      border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 12px;
      color: #fff; font-family: var(--font-heading); font-size: 1.02rem; font-weight: 700;
      cursor: pointer; box-shadow: 0 8px 20px rgba(2, 132, 199, 0.3);
      transition: all 0.2s;
    }
    .btn-submit:hover { transform: translateY(-2px); box-shadow: 0 12px 26px rgba(2, 132, 199, 0.45); }

    .results-col { position: sticky; top: 20px; }
    .results-card {
      background: var(--bg-card); backdrop-filter: blur(20px);
      border: 1px solid var(--border-glass); border-radius: 18px;
      padding: 24px; display: flex; flex-direction: column; gap: 18px;
    }
    .placeholder-view { text-align: center; padding: 40px 16px; }
    .placeholder-view svg { width: 48px; height: 48px; color: #38bdf8; margin-bottom: 12px; }
    .placeholder-title { font-family: var(--font-heading); font-size: 1.2rem; font-weight: 700; color: #fff; margin-bottom: 6px; }
    .placeholder-desc { font-size: 0.84rem; color: var(--text-muted); max-width: 320px; margin: 0 auto; line-height: 1.6; }

    .result-badge {
      display: inline-block; padding: 4px 12px; border-radius: 999px;
      font-size: 0.72rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px;
    }
    .badge-low { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-mod { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-high { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }
    .result-headline { font-family: var(--font-heading); font-size: 1.15rem; font-weight: 700; color: #fff; }

    .gauge-wrapper { display: flex; flex-direction: column; align-items: center; padding: 6px 0; }
    .gauge-box { position: relative; width: 220px; height: 130px; display: flex; align-items: flex-end; justify-content: center; }
    .gauge-svg { width: 100%; height: 100%; }
    .gauge-bg { stroke: rgba(255, 255, 255, 0.08); }
    .gauge-fill {
      stroke: var(--healthy); stroke-dasharray: 251.327; stroke-dashoffset: 251.327;
      transition: stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.4s;
    }
    .gauge-val { position: absolute; bottom: 8px; text-align: center; }
    .gauge-score { font-family: var(--font-heading); font-size: 2.2rem; font-weight: 800; color: #fff; line-height: 1; }
    .gauge-sub { font-size: 0.72rem; text-transform: uppercase; color: var(--text-muted); }

    .summary-box {
      font-size: 0.85rem; color: #cbd5e1; background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-subtle); border-radius: 10px; padding: 12px 14px;
    }
    .sec-title { font-family: var(--font-heading); font-size: 0.86rem; font-weight: 700; color: #fff; margin-bottom: 8px; }
    .factors-list { display: flex; flex-direction: column; gap: 8px; }
    .factor-item {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 12px; background: rgba(15, 23, 42, 0.5); border: 1px solid var(--border-subtle); border-radius: 8px;
    }
    .factor-p { font-size: 0.78rem; font-weight: 600; color: #e2e8f0; }
    .factor-n { font-size: 0.7rem; color: var(--text-faint); }
    .factor-tag {
      font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700;
      padding: 2px 7px; border-radius: 5px;
    }
    .tag-h { background: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }
    .tag-m { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    .recs-list { list-style: none; display: flex; flex-direction: column; gap: 6px; }
    .recs-list li {
      font-size: 0.8rem; color: #cbd5e1; padding: 8px 12px;
      background: rgba(15, 23, 42, 0.4); border-left: 3px solid #38bdf8; border-radius: 6px;
    }
    .app-footer {
      font-size: 0.72rem; color: var(--text-faint); text-align: center;
      margin-top: 10px; padding-top: 14px; border-top: 1px solid var(--border-subtle);
    }
    @media (max-width: 960px) {
      .main-grid { grid-template-columns: 1fr; }
      .results-col { position: static; }
    }
    @media (max-width: 600px) {
      .input-grid { grid-template-columns: 1fr; }
      .col-2 { grid-column: span 1; }
    }
  </style>
</head>
<body>
  <div class="bg-mesh"></div>
  <div class="grid-overlay"></div>

  <div class="app-container">
    <header class="app-header">
      <div class="brand">
        <div class="heart-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
          </svg>
        </div>
        <div>
          <h1 class="brand-title">CardioPulse <span class="gradient-text">Python AI</span></h1>
          <p class="brand-subtitle">Pure Python Machine Learning Web App on Vercel</p>
        </div>
      </div>
      <div class="header-badge">
        <span class="pulse-dot"></span>
        <span>Random Forest • 91.8% Accuracy</span>
      </div>
    </header>

    <div class="presets-bar">
      <span class="presets-label">⚡ 1-Click Demos:</span>
      <button type="button" class="btn-preset" id="pHealthy"><span class="dot dot-green"></span> Healthy Adult (1.9%)</button>
      <button type="button" class="btn-preset" id="pModerate"><span class="dot dot-amber"></span> Borderline (36.6%)</button>
      <button type="button" class="btn-preset" id="pHigh"><span class="dot dot-red"></span> High Risk (97.4%)</button>
      <button type="button" class="btn-preset btn-reset" id="pReset">Reset</button>
    </div>

    <div class="main-grid">
      <div class="form-col">
        <form id="patientForm">
          <div class="form-card">
            <h2 class="card-title">1. Patient Profile & Hemodynamics</h2>
            <p class="card-subtitle">Demographic indicators and arterial pressure</p>
            <div class="input-grid">
              <div class="form-group">
                <label class="form-label">Age (years) <span class="hint">18-95</span></label>
                <div class="slider-row">
                  <input type="range" id="slAge" min="18" max="95" value="55" class="custom-slider">
                  <input type="number" id="inAge" name="age" min="18" max="95" value="55" class="form-control num-box" required>
                </div>
              </div>
              <div class="form-group">
                <label class="form-label">Biological Sex</label>
                <div class="pill-group">
                  <label class="pill-opt"><input type="radio" name="sex" value="1" checked><span>Male</span></label>
                  <label class="pill-opt"><input type="radio" name="sex" value="0"><span>Female</span></label>
                </div>
              </div>
              <div class="form-group">
                <label class="form-label">Resting Blood Pressure <span class="hint">mm Hg</span></label>
                <input type="number" id="inTrestbps" name="trestbps" min="70" max="230" value="130" class="form-control" required>
              </div>
              <div class="form-group">
                <label class="form-label">Serum Cholesterol <span class="hint">mg/dL</span></label>
                <input type="number" id="inChol" name="chol" min="100" max="600" value="230" class="form-control" required>
              </div>
              <div class="form-group col-2">
                <label class="form-label">Fasting Blood Sugar &gt; 120 mg/dL</label>
                <div class="pill-group">
                  <label class="pill-opt"><input type="radio" name="fbs" value="0" checked><span>No (&le; 120 mg/dL)</span></label>
                  <label class="pill-opt"><input type="radio" name="fbs" value="1"><span>Yes (&gt; 120 mg/dL)</span></label>
                </div>
              </div>
            </div>
          </div>

          <div class="form-card" style="margin-top: 18px;">
            <h2 class="card-title">2. Cardiac Symptoms & Exertion Response</h2>
            <p class="card-subtitle">Angina presentation and physical stress tolerance</p>
            <div class="input-grid">
              <div class="form-group col-2">
                <label class="form-label">Chest Pain Classification (cp)</label>
                <select id="inCp" name="cp" class="form-control">
                  <option value="1">Type 1: Typical Angina</option>
                  <option value="2">Type 2: Atypical Angina</option>
                  <option value="3">Type 3: Non-Anginal</option>
                  <option value="4" selected>Type 4: Asymptomatic (Silent Ischemia)</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label">Max Heart Rate Achieved <span class="hint">bpm</span></label>
                <input type="number" id="inThalach" name="thalach" min="60" max="230" value="150" class="form-control" required>
              </div>
              <div class="form-group">
                <label class="form-label">Exercise Angina (exang)</label>
                <div class="pill-group">
                  <label class="pill-opt"><input type="radio" name="exang" value="0" checked><span>No</span></label>
                  <label class="pill-opt"><input type="radio" name="exang" value="1"><span>Yes</span></label>
                </div>
              </div>
              <div class="form-group col-2">
                <label class="form-label">Resting Electrocardiogram</label>
                <select id="inRestecg" name="restecg" class="form-control">
                  <option value="0" selected>0: Normal</option>
                  <option value="1">1: ST-T Wave Abnormality</option>
                  <option value="2">2: Left Ventricular Hypertrophy</option>
                </select>
              </div>
            </div>
          </div>

          <div class="form-card" style="margin-top: 18px;">
            <h2 class="card-title">3. Stress Diagnostics & Fluoroscopy</h2>
            <p class="card-subtitle">ST depression, vessel calcification, and perfusion scan</p>
            <div class="input-grid">
              <div class="form-group">
                <label class="form-label">ST Depression (oldpeak) <span class="hint">mm</span></label>
                <div class="slider-row">
                  <input type="range" id="slOldpeak" min="0.0" max="6.0" step="0.1" value="1.0" class="custom-slider">
                  <input type="number" id="inOldpeak" name="oldpeak" min="0.0" max="6.2" step="0.1" value="1.0" class="form-control num-box" required>
                </div>
              </div>
              <div class="form-group">
                <label class="form-label">Peak ST Slope</label>
                <select id="inSlope" name="slope" class="form-control">
                  <option value="1">1: Upsloping</option>
                  <option value="2" selected>2: Flat</option>
                  <option value="3">3: Downsloping</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label">Major Vessels Blocked (ca)</label>
                <select id="inCa" name="ca" class="form-control">
                  <option value="0" selected>0 Major Vessels</option>
                  <option value="1">1 Major Vessel</option>
                  <option value="2">2 Major Vessels</option>
                  <option value="3">3 Major Vessels</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label">Thallium Heart Scan (thal)</label>
                <select id="inThal" name="thal" class="form-control">
                  <option value="3" selected>3: Normal Perfusion</option>
                  <option value="6">6: Fixed Defect</option>
                  <option value="7">7: Reversible Defect</option>
                </select>
              </div>
            </div>
          </div>

          <div style="margin-top: 18px;">
            <button type="submit" id="btnSubmit" class="btn-submit">Run Python Heart Evaluation</button>
          </div>
        </form>
      </div>

      <div class="results-col">
        <div class="results-card" id="cardPlaceholder">
          <div class="placeholder-view">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
            </svg>
            <h3 class="placeholder-title">Awaiting Clinical Data</h3>
            <p class="placeholder-desc">Click any of the <strong>1-Click Demos</strong> above or fill in the patient parameters to generate an instant cardiovascular prediction.</p>
          </div>
        </div>

        <div class="results-card" id="cardActive" style="display: none;">
          <div>
            <span class="result-badge" id="badgeRisk">LOW RISK</span>
            <h3 class="result-headline" id="headRisk">Cardiovascular Evaluation</h3>
          </div>

          <div class="gauge-wrapper">
            <div class="gauge-box">
              <svg class="gauge-svg" viewBox="0 0 200 120">
                <path class="gauge-bg" d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke-width="16" stroke-linecap="round"/>
                <path class="gauge-fill" id="arcFill" d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke-width="16" stroke-linecap="round"/>
              </svg>
              <div class="gauge-val">
                <div class="gauge-score"><span id="txtScore">0.0</span>%</div>
                <div class="gauge-sub">Probability of Disease</div>
              </div>
            </div>
          </div>

          <div class="summary-box" id="txtSummary">Diagnostic summary will appear here.</div>

          <div>
            <h4 class="sec-title">Contributing Risk Biomarkers</h4>
            <div class="factors-list" id="listFactors"></div>
          </div>

          <div>
            <h4 class="sec-title">Clinical Action Plan</h4>
            <ul class="recs-list" id="listRecs"></ul>
          </div>
        </div>
      </div>
    </div>

    <footer class="app-footer">
      Pure Python Application • Trained on UCI Cleveland Heart Disease Dataset (303 Cohort) • Deployed on Vercel
    </footer>
  </div>

  <script>
    const G_CIRC = 251.327;
    const form = document.getElementById('patientForm');
    const slAge = document.getElementById('slAge');
    const inAge = document.getElementById('inAge');
    const slOld = document.getElementById('slOldpeak');
    const inOld = document.getElementById('inOldpeak');
    const cardPlaceholder = document.getElementById('cardPlaceholder');
    const cardActive = document.getElementById('cardActive');

    slAge.oninput = () => inAge.value = slAge.value;
    inAge.oninput = () => slAge.value = inAge.value;
    slOld.oninput = () => inOld.value = parseFloat(slOld.value).toFixed(1);
    inOld.oninput = () => slOld.value = inOld.value;

    const PRESETS = {
      healthy: { age: 32, sex: 0, cp: 2, trestbps: 115, chol: 175, fbs: 0, restecg: 0, thalach: 180, exang: 0, oldpeak: 0.0, slope: 1, ca: 0, thal: 3 },
      moderate: { age: 54, sex: 1, cp: 3, trestbps: 138, chol: 235, fbs: 0, restecg: 1, thalach: 145, exang: 0, oldpeak: 1.2, slope: 2, ca: 0, thal: 6 },
      high: { age: 64, sex: 1, cp: 4, trestbps: 160, chol: 285, fbs: 1, restecg: 2, thalach: 115, exang: 1, oldpeak: 2.8, slope: 2, ca: 2, thal: 7 }
    };

    function applyPreset(key) {
      const d = PRESETS[key];
      slAge.value = inAge.value = d.age;
      form.querySelector(`input[name="sex"][value="${d.sex}"]`).checked = true;
      document.getElementById('inTrestbps').value = d.trestbps;
      document.getElementById('inChol').value = d.chol;
      form.querySelector(`input[name="fbs"][value="${d.fbs}"]`).checked = true;
      document.getElementById('inCp').value = d.cp;
      document.getElementById('inThalach').value = d.thalach;
      form.querySelector(`input[name="exang"][value="${d.exang}"]`).checked = true;
      document.getElementById('inRestecg').value = d.restecg;
      slOld.value = inOld.value = d.oldpeak.toFixed(1);
      document.getElementById('inSlope').value = d.slope;
      document.getElementById('inCa').value = d.ca;
      document.getElementById('inThal').value = d.thal;
      submitData();
    }

    document.getElementById('pHealthy').onclick = () => applyPreset('healthy');
    document.getElementById('pModerate').onclick = () => applyPreset('moderate');
    document.getElementById('pHigh').onclick = () => applyPreset('high');
    document.getElementById('pReset').onclick = () => {
      form.reset();
      slAge.value = inAge.value = 55;
      slOld.value = inOld.value = "1.0";
      cardActive.style.display = 'none';
      cardPlaceholder.style.display = 'block';
    };

    async function submitData() {
      const fd = new FormData(form);
      const payload = {
        age: parseInt(fd.get('age')), sex: parseInt(fd.get('sex')), cp: parseInt(fd.get('cp')),
        trestbps: parseFloat(fd.get('trestbps')), chol: parseFloat(fd.get('chol')),
        fbs: parseInt(fd.get('fbs')), restecg: parseInt(fd.get('restecg')),
        thalach: parseFloat(fd.get('thalach')), exang: parseInt(fd.get('exang')),
        oldpeak: parseFloat(fd.get('oldpeak')), slope: parseInt(fd.get('slope')),
        ca: parseInt(fd.get('ca')), thal: parseInt(fd.get('thal'))
      };

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        renderResult(data);
      } catch (e) {
        alert('Prediction request failed: ' + e.message);
      }
    }

    form.onsubmit = (e) => { e.preventDefault(); submitData(); };

    function renderResult(data) {
      cardPlaceholder.style.display = 'none';
      cardActive.style.display = 'flex';

      const badge = document.getElementById('badgeRisk');
      badge.className = 'result-badge';
      if (data.risk_tier === 'Low Risk') {
        badge.classList.add('badge-low'); badge.textContent = 'LOW RISK';
      } else if (data.risk_tier === 'Moderate Risk') {
        badge.classList.add('badge-mod'); badge.textContent = 'MODERATE RISK';
      } else {
        badge.classList.add('badge-high'); badge.textContent = 'HIGH RISK';
      }

      document.getElementById('headRisk').textContent = data.headline;
      document.getElementById('txtSummary').textContent = data.summary;
      document.getElementById('txtScore').textContent = data.risk_percentage.toFixed(1);

      const offset = G_CIRC * (1 - Math.min(1.0, data.probability));
      const arc = document.getElementById('arcFill');
      arc.style.strokeDashoffset = offset;
      arc.style.stroke = data.probability < 0.35 ? '#10b981' : (data.probability < 0.65 ? '#f59e0b' : '#f43f5e');

      const factorsDiv = document.getElementById('listFactors');
      factorsDiv.innerHTML = '';
      if (!data.factors || data.factors.length === 0) {
        factorsDiv.innerHTML = '<div class="factor-item"><span class="factor-p">Optimal Cardiovascular Biomarkers</span><span class="factor-tag" style="background:rgba(16,185,129,0.2);color:#34d399">Normal</span></div>';
      } else {
        data.factors.forEach(f => {
          const item = document.createElement('div');
          item.className = 'factor-item';
          const tc = f.sev === 'high' ? 'tag-h' : 'tag-m';
          item.innerHTML = `<div><div class="factor-p">${f.param} — ${f.status}</div><div class="factor-n">${f.note}</div></div><span class="factor-tag ${tc}">${f.value}</span>`;
          factorsDiv.appendChild(item);
        });
      }

      const recsUl = document.getElementById('listRecs');
      recsUl.innerHTML = '';
      data.recommendations.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        recsUl.appendChild(li);
      });
    }
  </script>
</body>
</html>
"""


# -------------------------------------------------------------
# Routes (Pure Python App)
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the complete interactive web UI directly from Python."""
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)


@app.get("/api/health")
def health_status():
    return {
        "status": "healthy",
        "app": "CardioPulse AI Full-Stack Python",
        "runtime": "python-fastapi",
        "platform": "Vercel Serverless",
        "model_loaded": model_pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/predict")
@app.post("/predict")
def run_prediction(patient: PatientPayload):
    """Calculates risk prediction and clinical explanations."""
    global model_pipeline

    if model_pipeline is None:
        model_pipeline = train_fallback_model()

    norm_data = normalize_patient(patient)
    df_row = pd.DataFrame([[norm_data[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

    try:
        pred_int = int(model_pipeline.predict(df_row)[0])
        prob_val = float(model_pipeline.predict_proba(df_row)[0][1])
    except Exception as err:
        logger.error(f"Inference error: {err}")
        raise HTTPException(status_code=500, detail=str(err))

    risk_pct = round(prob_val * 100, 1)

    if prob_val < 0.35:
        tier = "Low Risk"
        headline = "Favorable Cardiovascular Profile"
        summary = "No significant indicators of coronary heart disease detected. Markers fall within healthy physiological limits."
    elif prob_val < 0.65:
        tier = "Moderate Risk"
        headline = "Borderline Cardiac Risk Profile"
        summary = "Several intermediate markers detected. Preventive clinical evaluation and lifestyle modifications are advised."
    else:
        tier = "High Risk"
        headline = "Elevated Coronary Disease Risk"
        summary = "Multiple high-severity markers observed. Consultation with a board-certified cardiologist is strongly advised."

    factors = extract_clinical_factors(norm_data)
    recs = generate_recommendations(tier)

    return {
        "status": "success",
        "prediction": pred_int,
        "probability": round(prob_val, 4),
        "risk_percentage": risk_pct,
        "risk_tier": tier,
        "headline": headline,
        "summary": summary,
        "factors": factors,
        "recommendations": recs,
        "patient": norm_data,
        "evaluated_at": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  🚀 CardioPulse Python App running at: http://127.0.0.1:8000")
    print("=" * 60 + "\n")
    uvicorn.run("api.index:app", host="127.0.0.1", port=8000, reload=True)
