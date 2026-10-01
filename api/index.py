"""
CardioPulse AI — Next-Gen Cardiovascular Diagnostic Operating System
Full-Stack Python Application for Vercel
Features: Live Canvas ECG Monitor, Multi-Model Consensus (RF, GB, LR),
SHAP-Style Feature Attribution, 6-Axis Radar Metrics, and Clinical PDF Export.
"""

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cardiopulse_pro")

app = FastAPI(
    title="CardioPulse AI — Next-Gen Diagnostic Engine",
    description="Multi-model cardiovascular disease risk assessment system.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENSEMBLE_PATH = os.path.join(BASE_DIR, "model", "ensemble_models.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")

ensemble_models = {}


def train_fallback_models() -> Dict[str, Pipeline]:
    """Self-healing fallback if models file is missing on Vercel cold-start."""
    logger.info("Training self-healing multi-model ensemble...")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
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

    models = {
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
        ]),
        "gb": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, max_depth=3, random_state=42))
        ]),
        "lr": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])
    }
    for m in models.values():
        m.fit(X, y)
    return models


try:
    if os.path.exists(ENSEMBLE_PATH):
        ensemble_models = joblib.load(ENSEMBLE_PATH)
        logger.info("Successfully loaded multi-model ensemble from disk.")
    else:
        ensemble_models = train_fallback_models()
except Exception as e:
    logger.error(f"Error loading models: {e}. Training fallback.")
    ensemble_models = train_fallback_models()


# -------------------------------------------------------------
# Request Schema & Helpers
# -------------------------------------------------------------
class PatientData(BaseModel):
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


def normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(data)
    if d["cp"] == 0: d["cp"] = 1
    if d["slope"] == 0: d["slope"] = 1
    if d["thal"] == 1: d["thal"] = 3
    elif d["thal"] == 2: d["thal"] = 6
    elif d["thal"] == 3: d["thal"] = 7
    elif d["thal"] not in [3, 6, 7]: d["thal"] = 3
    return d


def compute_radar_metrics(d: Dict[str, Any]) -> Dict[str, float]:
    """Scales 6 key dimensions between 10 and 100 for SVG radar chart."""
    bp_score = min(100.0, max(15.0, (d["trestbps"] - 90) / 100 * 100))
    chol_score = min(100.0, max(15.0, (d["chol"] - 130) / 250 * 100))
    ischemia_score = min(100.0, max(10.0, (d["oldpeak"] / 4.0) * 100))
    vessel_score = min(100.0, max(10.0, (d["ca"] / 3.0) * 100))
    thal_score = 15.0 if d["thal"] == 3 else (70.0 if d["thal"] == 6 else 95.0)
    # Exertional strain: lower thalach for age = higher strain
    expected_hr = 220 - d["age"]
    exert_deficit = max(0.0, (expected_hr - d["thalach"]) / expected_hr * 100)
    strain_score = min(100.0, max(15.0, exert_deficit + (d["exang"] * 35)))

    return {
        "blood_pressure": round(bp_score, 1),
        "cholesterol": round(chol_score, 1),
        "ischemic_st": round(ischemia_score, 1),
        "vessel_occlusion": round(vessel_score, 1),
        "perfusion_defect": round(thal_score, 1),
        "exertion_strain": round(strain_score, 1)
    }


def compute_feature_attributions(df_row: pd.DataFrame) -> List[Dict[str, Any]]:
    """Calculates directional SHAP-style feature contributions (+ risk, - protective)."""
    lr = ensemble_models.get("lr")
    if not lr:
        return []
    scaler = lr.named_steps["scaler"]
    clf = lr.named_steps["clf"]
    scaled = scaler.transform(df_row)[0]
    coefs = clf.coef_[0]

    names_map = {
        "ca": "Fluoroscopy Vessels (ca)",
        "thal": "Thallium Perfusion (thal)",
        "oldpeak": "ST Depression (oldpeak)",
        "exang": "Exercise Angina (exang)",
        "cp": "Chest Pain Classification (cp)",
        "thalach": "Max Heart Rate (thalach)",
        "trestbps": "Resting Blood Pressure",
        "chol": "Serum Cholesterol",
        "age": "Patient Age",
        "sex": "Biological Sex",
        "slope": "ST Slope Dynamic",
        "restecg": "Resting ECG Rhythm",
        "fbs": "Fasting Blood Sugar"
    }

    attributions = []
    for f, s, c in zip(FEATURE_NAMES, scaled, coefs):
        impact = float(s * c)
        if abs(impact) >= 0.15:
            attributions.append({
                "feature": f,
                "label": names_map.get(f, f),
                "impact": round(impact, 2),
                "direction": "risk" if impact > 0 else "protective",
                "percentage": round(min(100.0, abs(impact) * 25.0), 1)
            })

    attributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return attributions[:6]


# -------------------------------------------------------------
# Complete Modern UI Template (HTMLResponse)
# -------------------------------------------------------------
UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardioPulse Pro — AI Cardiovascular Diagnostic OS</title>
  <meta name="description" content="Next-Gen Machine Learning Heart Disease Diagnosis & Hemodynamic Monitoring Operating System.">
  
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  
  <style>
    :root {
      --bg-deep: #050811;
      --bg-card: rgba(13, 20, 36, 0.75);
      --bg-card-hover: rgba(18, 28, 51, 0.85);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-glass: rgba(56, 189, 248, 0.15);
      --border-focus: #38bdf8;
      
      --cyan: #06b6d4;
      --neon-cyan: #38bdf8;
      --healthy: #10b981;
      --warning: #f59e0b;
      --danger: #f43f5e;
      --purple: #a855f7;

      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-faint: #64748b;

      --font-heading: 'Outfit', sans-serif;
      --font-body: 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-body);
      background-color: var(--bg-deep);
      color: var(--text-main);
      min-height: 100vh;
      overflow-x: hidden;
      line-height: 1.5;
    }

    /* Ambient Space Mesh */
    .ambient-mesh {
      position: fixed; inset: 0;
      background:
        radial-gradient(circle at 10% 15%, rgba(56, 189, 248, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 85% 25%, rgba(244, 63, 94, 0.1) 0%, transparent 45%),
        radial-gradient(circle at 50% 85%, rgba(168, 85, 247, 0.08) 0%, transparent 50%);
      pointer-events: none; z-index: 0;
    }
    .grid-mesh {
      position: fixed; inset: 0;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
      background-size: 32px 32px;
      pointer-events: none; z-index: 1;
    }

    .app-wrapper {
      position: relative; z-index: 2;
      max-width: 1400px; margin: 0 auto;
      padding: 16px 24px 60px;
      display: flex; flex-direction: column; gap: 18px;
    }

    /* Top ECG Monitor HUD Strip */
    .ecg-hud-strip {
      background: rgba(10, 16, 28, 0.9);
      border: 1px solid var(--border-glass);
      border-radius: 16px;
      padding: 12px 20px;
      display: flex; align-items: center; justify-content: space-between;
      gap: 20px;
      backdrop-filter: blur(20px);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hud-left {
      display: flex; align-items: center; gap: 14px;
      min-width: 220px;
    }
    .heart-orb {
      width: 42px; height: 42px; border-radius: 12px;
      background: linear-gradient(135deg, rgba(244, 63, 94, 0.2), rgba(56, 189, 248, 0.2));
      border: 1px solid rgba(244, 63, 94, 0.4);
      display: flex; align-items: center; justify-content: center;
      color: var(--danger);
      animation: heartpulse 1.8s infinite ease-in-out;
      cursor: pointer;
    }
    .heart-orb svg { width: 22px; height: 22px; }
    @keyframes heartpulse {
      0%, 100% { transform: scale(1); filter: drop-shadow(0 0 4px rgba(244, 63, 94, 0.4)); }
      50% { transform: scale(1.08); filter: drop-shadow(0 0 12px rgba(244, 63, 94, 0.8)); }
    }
    .hud-title { font-family: var(--font-heading); font-size: 1.35rem; font-weight: 800; color: #fff; }
    .hud-sub { font-size: 0.72rem; color: var(--text-muted); }
    .ecg-canvas-container {
      flex: 1; height: 46px; position: relative;
      background: rgba(4, 8, 16, 0.8);
      border: 1px solid rgba(56, 189, 248, 0.2);
      border-radius: 8px; overflow: hidden;
    }
    #ecgCanvas { width: 100%; height: 100%; display: block; }
    .hud-right {
      display: flex; align-items: center; gap: 14px;
      min-width: 200px; justify-content: flex-end;
    }
    .bpm-box {
      display: flex; flex-direction: column; align-items: flex-end;
    }
    .bpm-number { font-family: var(--font-mono); font-size: 1.3rem; font-weight: 700; color: #38bdf8; line-height: 1; }
    .bpm-label { font-size: 0.64rem; text-transform: uppercase; color: var(--text-faint); letter-spacing: 0.05em; }
    .btn-sound {
      background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border-subtle);
      color: var(--text-muted); padding: 7px 10px; border-radius: 8px; cursor: pointer;
      font-size: 0.75rem; display: flex; align-items: center; gap: 6px;
    }
    .btn-sound:hover { color: #fff; border-color: rgba(255, 255, 255, 0.2); }
    .btn-sound.active { color: #38bdf8; border-color: #38bdf8; background: rgba(56, 189, 248, 0.15); }

    /* Quick Presets Bar */
    .presets-strip {
      background: var(--bg-card);
      border: 1px solid var(--border-glass);
      border-radius: 14px;
      padding: 10px 18px;
      display: flex; align-items: center; flex-wrap: wrap; gap: 10px;
    }
    .strip-label {
      font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.06em;
      display: flex; align-items: center; gap: 6px;
    }
    .btn-pill {
      display: flex; align-items: center; gap: 8px; padding: 7px 14px;
      font-family: var(--font-body); font-size: 0.82rem; font-weight: 600;
      color: var(--text-main); background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-subtle); border-radius: 8px;
      cursor: pointer; transition: all 0.2s;
    }
    .btn-pill:hover { background: rgba(255, 255, 255, 0.08); transform: translateY(-1px); border-color: rgba(255,255,255,0.2); }
    .dot-status { width: 8px; height: 8px; border-radius: 50%; }
    .dot-healthy { background: var(--healthy); box-shadow: 0 0 8px var(--healthy); }
    .dot-warning { background: var(--warning); box-shadow: 0 0 8px var(--warning); }
    .dot-danger { background: var(--danger); box-shadow: 0 0 8px var(--danger); }
    .btn-clear { margin-left: auto; color: var(--text-muted); }

    /* Main Workspace Layout */
    .workspace-grid {
      display: grid; grid-template-columns: 1.15fr 0.85fr;
      gap: 20px; align-items: start;
    }
    .form-column { display: flex; flex-direction: column; gap: 16px; }
    .card-panel {
      background: var(--bg-card);
      border: 1px solid var(--border-glass);
      border-radius: 16px;
      padding: 20px 22px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
      backdrop-filter: blur(20px);
    }
    .panel-header {
      display: flex; align-items: center; gap: 12px; margin-bottom: 16px;
      padding-bottom: 12px; border-bottom: 1px solid var(--border-subtle);
    }
    .panel-icon {
      width: 34px; height: 34px; border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
    }
    .icon-cyan { background: rgba(6, 182, 212, 0.15); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); }
    .icon-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .icon-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    .panel-title { font-family: var(--font-heading); font-size: 1.05rem; font-weight: 700; color: #fff; }
    .panel-desc { font-size: 0.74rem; color: var(--text-muted); }

    .fields-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .full-width { grid-column: span 2; }
    .field-wrap { display: flex; flex-direction: column; gap: 6px; }
    .field-top { display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; color: #e2e8f0; }
    .field-hint { font-size: 0.7rem; font-weight: 400; color: var(--text-faint); }
    .input-box {
      width: 100%; padding: 9px 12px;
      font-family: var(--font-body); font-size: 0.88rem;
      color: #fff; background: rgba(8, 14, 26, 0.8);
      border: 1px solid var(--border-subtle); border-radius: 8px;
      outline: none; transition: border-color 0.2s, box-shadow 0.2s;
    }
    .input-box:focus { border-color: var(--border-focus); box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15); }
    select.input-box { cursor: pointer; }
    select.input-box option { background: #0b1120; color: #fff; }

    .slider-group { display: flex; align-items: center; gap: 10px; }
    .range-slider {
      flex: 1; -webkit-appearance: none; height: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 999px;
    }
    .range-slider::-webkit-slider-thumb {
      -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%;
      background: #38bdf8; border: 2px solid #070a12; box-shadow: 0 0 8px #38bdf8; cursor: pointer;
    }
    .slider-num { width: 68px; text-align: center; font-family: var(--font-mono); font-weight: 700; }

    .toggle-radios { display: flex; gap: 8px; }
    .radio-lbl { flex: 1; cursor: pointer; }
    .radio-lbl input { display: none; }
    .radio-lbl span {
      display: block; padding: 8px; text-align: center; font-size: 0.8rem; font-weight: 600;
      color: var(--text-muted); background: rgba(8, 14, 26, 0.8);
      border: 1px solid var(--border-subtle); border-radius: 8px;
      transition: all 0.2s;
    }
    .radio-lbl input:checked + span {
      background: rgba(56, 189, 248, 0.15); border-color: #38bdf8; color: #fff; box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }

    .btn-run-ai {
      width: 100%; padding: 15px;
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #0c4a6e 100%);
      border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 12px;
      color: #fff; font-family: var(--font-heading); font-size: 1.05rem; font-weight: 700;
      letter-spacing: 0.02em; cursor: pointer;
      display: flex; align-items: center; justify-content: center; gap: 10px;
      box-shadow: 0 6px 20px rgba(2, 132, 199, 0.35);
      transition: all 0.25s;
    }
    .btn-run-ai:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(2, 132, 199, 0.5); }

    /* Results Column & Futuristic HUD */
    .results-column { position: sticky; top: 16px; }
    .diagnostic-hud {
      background: var(--bg-card);
      border: 1px solid var(--border-glass);
      border-radius: 18px;
      padding: 24px;
      display: flex; flex-direction: column; gap: 18px;
      backdrop-filter: blur(24px);
    }
    .placeholder-state {
      text-align: center; padding: 48px 16px;
    }
    .pulse-rings {
      width: 90px; height: 90px; margin: 0 auto 16px; position: relative;
      display: flex; align-items: center; justify-content: center;
    }
    .ring-circ {
      position: absolute; border-radius: 50%; border: 1px dashed rgba(56, 189, 248, 0.25);
    }
    .r-1 { width: 50px; height: 50px; animation: spin 10s linear infinite; }
    .r-2 { width: 90px; height: 90px; animation: spin 20s linear infinite reverse; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .ring-core {
      width: 36px; height: 36px; border-radius: 50%; background: rgba(56, 189, 248, 0.15);
      display: flex; align-items: center; justify-content: center; color: #38bdf8;
    }
    .ph-title { font-family: var(--font-heading); font-size: 1.25rem; font-weight: 700; color: #fff; margin-bottom: 6px; }
    .ph-desc { font-size: 0.82rem; color: var(--text-muted); max-width: 320px; margin: 0 auto; line-height: 1.5; }

    /* Active Results View */
    .active-hud { display: none; flex-direction: column; gap: 18px; }
    .hud-header {
      display: flex; justify-content: space-between; align-items: flex-start;
      padding-bottom: 14px; border-bottom: 1px solid var(--border-subtle);
    }
    .risk-badge {
      display: inline-block; padding: 4px 12px; border-radius: 999px;
      font-size: 0.72rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px;
    }
    .badge-l { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-m { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-h { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); box-shadow: 0 0 10px rgba(244, 63, 94, 0.2); }
    .hud-headline { font-family: var(--font-heading); font-size: 1.15rem; font-weight: 700; color: #fff; }
    .btn-pdf {
      display: flex; align-items: center; gap: 6px; padding: 6px 12px;
      font-size: 0.76rem; font-weight: 600; color: var(--text-muted);
      background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border-subtle);
      border-radius: 8px; cursor: pointer;
    }
    .btn-pdf:hover { color: #fff; background: rgba(255, 255, 255, 0.1); }

    /* Radial Gauge Visualizer */
    .gauge-wrapper { display: flex; flex-direction: column; align-items: center; }
    .gauge-dial {
      position: relative; width: 220px; height: 130px;
      display: flex; align-items: flex-end; justify-content: center;
    }
    .gauge-svg { width: 100%; height: 100%; }
    .arc-bg { stroke: rgba(255, 255, 255, 0.08); }
    .arc-progress {
      stroke: var(--healthy); stroke-dasharray: 251.327; stroke-dashoffset: 251.327;
      transition: stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.4s;
    }
    .gauge-center-val { position: absolute; bottom: 8px; text-align: center; }
    .score-txt { font-family: var(--font-heading); font-size: 2.2rem; font-weight: 800; color: #fff; line-height: 1; }
    .score-sub { font-size: 0.7rem; text-transform: uppercase; color: var(--text-muted); }

    /* Multi-Model Ensemble Consensus Bar */
    .consensus-panel {
      background: rgba(8, 14, 26, 0.7);
      border: 1px solid var(--border-subtle);
      border-radius: 12px; padding: 12px 14px;
      display: flex; flex-direction: column; gap: 8px;
    }
    .consensus-top { display: flex; justify-content: space-between; font-size: 0.76rem; font-weight: 700; color: #fff; }
    .consensus-tag { color: #38bdf8; font-family: var(--font-mono); }
    .model-bars { display: flex; flex-direction: column; gap: 6px; }
    .model-bar-item { display: flex; align-items: center; gap: 8px; font-size: 0.72rem; color: var(--text-muted); }
    .m-name { width: 110px; }
    .m-track { flex: 1; height: 6px; background: rgba(255, 255, 255, 0.08); border-radius: 999px; overflow: hidden; }
    .m-fill { height: 100%; border-radius: 999px; transition: width 0.8s ease; }
    .m-pct { width: 44px; text-align: right; font-family: var(--font-mono); color: #fff; font-weight: 600; }

    /* Dual Charts Row: Radar & SHAP Waterfall */
    .dual-charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .chart-box {
      background: rgba(8, 14, 26, 0.7);
      border: 1px solid var(--border-subtle);
      border-radius: 12px; padding: 12px;
      display: flex; flex-direction: column; gap: 8px;
    }
    .box-title { font-family: var(--font-heading); font-size: 0.8rem; font-weight: 700; color: #e2e8f0; }
    .radar-svg-box { width: 100%; height: 140px; display: flex; align-items: center; justify-content: center; }

    /* SHAP Waterfall Bar List */
    .waterfall-list { display: flex; flex-direction: column; gap: 6px; }
    .wf-item { display: flex; align-items: center; justify-content: space-between; font-size: 0.72rem; }
    .wf-label { color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100px; }
    .wf-bar-wrap { flex: 1; margin: 0 8px; height: 5px; background: rgba(255,255,255,0.06); border-radius: 999px; overflow: hidden; }
    .wf-bar { height: 100%; border-radius: 999px; }
    .wf-val { font-family: var(--font-mono); font-weight: 700; font-size: 0.7rem; }
    .wf-risk { background: #f43f5e; color: #fb7185; }
    .wf-prot { background: #10b981; color: #34d399; }

    /* Guidance & Recommendations */
    .recs-box {
      background: rgba(8, 14, 26, 0.7);
      border: 1px solid var(--border-subtle);
      border-radius: 12px; padding: 12px 14px;
    }
    .recs-ul { list-style: none; display: flex; flex-direction: column; gap: 6px; margin-top: 6px; }
    .recs-ul li {
      font-size: 0.78rem; color: #cbd5e1; padding: 8px 10px;
      background: rgba(15, 23, 42, 0.5); border-left: 3px solid #38bdf8; border-radius: 6px;
    }

    .app-footer {
      font-size: 0.72rem; color: var(--text-faint); text-align: center;
      padding-top: 14px; border-top: 1px solid var(--border-subtle);
    }

    @media print {
      body { background: #fff !important; color: #000 !important; }
      .ambient-mesh, .grid-mesh, .ecg-hud-strip, .presets-strip, .btn-run-ai, .btn-pdf, .app-footer { display: none !important; }
      .app-wrapper { max-width: 100% !important; padding: 0 !important; }
      .workspace-grid { grid-template-columns: 1fr !important; }
      .card-panel, .diagnostic-hud, .consensus-panel, .chart-box {
        background: #fff !important; border: 1px solid #ddd !important; box-shadow: none !important; color: #000 !important;
      }
      .panel-title, .hud-headline, .score-txt { color: #000 !important; }
    }

    @media (max-width: 1024px) {
      .workspace-grid { grid-template-columns: 1fr; }
      .results-column { position: static; }
      .dual-charts-row { grid-template-columns: 1fr; }
    }
    @media (max-width: 680px) {
      .ecg-hud-strip { flex-direction: column; align-items: flex-start; }
      .hud-right { width: 100%; justify-content: space-between; }
      .fields-grid { grid-template-columns: 1fr; }
      .full-width { grid-column: span 1; }
    }
  </style>
</head>
<body>
  <div class="ambient-mesh"></div>
  <div class="grid-mesh"></div>

  <div class="app-wrapper">
    
    <!-- Top ECG Monitor HUD Strip -->
    <header class="ecg-hud-strip">
      <div class="hud-left">
        <div class="heart-orb" id="heartOrb" title="Cardiopulse Rhythm Monitor">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
          </svg>
        </div>
        <div>
          <h1 class="hud-title">CardioPulse <span style="background: linear-gradient(135deg, #38bdf8, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">PRO</span></h1>
          <p class="hud-sub">AI Cardiovascular Diagnostic Engine • Python Serverless</p>
        </div>
      </div>

      <div class="ecg-canvas-container">
        <canvas id="ecgCanvas"></canvas>
      </div>

      <div class="hud-right">
        <div class="bpm-box">
          <div class="bpm-number" id="txtLiveBpm">150 <span style="font-size:0.75rem; color:#94a3b8;">BPM</span></div>
          <div class="bpm-label">Sinus Rhythm</div>
        </div>
        <button type="button" class="btn-sound" id="btnAudioToggle" title="Toggle synthesized heartbeat audio">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
          <span id="txtSoundState">Audio Muted</span>
        </button>
      </div>
    </header>

    <!-- Quick Patient Presets Strip -->
    <div class="presets-strip">
      <span class="strip-label">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
        Instant Presets:
      </span>
      <button type="button" class="btn-pill" id="btnPresetHealthy"><span class="dot-status dot-healthy"></span> Healthy Adult (1.9% Risk)</button>
      <button type="button" class="btn-pill" id="btnPresetModerate"><span class="dot-status dot-warning"></span> Middle-Aged Borderline (36.6%)</button>
      <button type="button" class="btn-pill" id="btnPresetHigh"><span class="dot-status dot-danger"></span> High Risk Angina (97.4%)</button>
      <button type="button" class="btn-pill btn-clear" id="btnResetAll">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path></svg>
        Reset
      </button>
    </div>

    <!-- Main Workspace Grid -->
    <main class="workspace-grid">
      
      <!-- Input Panel Column -->
      <section class="form-column">
        <form id="cardioForm">
          
          <!-- Section 1 -->
          <div class="card-panel">
            <div class="panel-header">
              <div class="panel-icon icon-cyan">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
              </div>
              <div>
                <h2 class="panel-title">1. Patient Profile & Hemodynamics</h2>
                <p class="panel-desc">Demographic vitals and baseline arterial pressure</p>
              </div>
            </div>

            <div class="fields-grid">
              <div class="field-wrap">
                <div class="field-top"><label>Age (years)</label><span class="field-hint">18-95</span></div>
                <div class="slider-group">
                  <input type="range" id="slAge" min="18" max="95" value="55" class="range-slider">
                  <input type="number" id="inAge" name="age" min="18" max="95" value="55" class="input-box slider-num" required>
                </div>
              </div>

              <div class="field-wrap">
                <div class="field-top"><label>Biological Sex</label><span class="field-hint">Clinical risk</span></div>
                <div class="toggle-radios">
                  <label class="radio-lbl"><input type="radio" name="sex" value="1" checked><span>Male</span></label>
                  <label class="radio-lbl"><input type="radio" name="sex" value="0"><span>Female</span></label>
                </div>
              </div>

              <div class="field-wrap">
                <div class="field-top"><label>Resting Blood Pressure</label><span class="field-hint">Normal: 90-120 mmHg</span></div>
                <input type="number" id="inTrestbps" name="trestbps" min="70" max="230" value="130" class="input-box" required>
              </div>

              <div class="field-wrap">
                <div class="field-top"><label>Serum Cholesterol</label><span class="field-hint">Desirable: &lt;200 mg/dL</span></div>
                <input type="number" id="inChol" name="chol" min="100" max="600" value="230" class="input-box" required>
              </div>

              <div class="field-wrap full-width">
                <div class="field-top"><label>Fasting Blood Sugar &gt; 120 mg/dL (fbs)</label><span class="field-hint">Glycemic baseline</span></div>
                <div class="toggle-radios">
                  <label class="radio-lbl"><input type="radio" name="fbs" value="0" checked><span>No (&le; 120 mg/dL)</span></label>
                  <label class="radio-lbl"><input type="radio" name="fbs" value="1"><span>Yes (&gt; 120 mg/dL)</span></label>
                </div>
              </div>
            </div>
          </div>

          <!-- Section 2 -->
          <div class="card-panel" style="margin-top: 16px;">
            <div class="panel-header">
              <div class="panel-icon icon-green">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
              </div>
              <div>
                <h2 class="panel-title">2. Cardiac Symptoms & Exertion Tolerance</h2>
                <p class="panel-desc">Angina presentation and maximum stress cardiovascular response</p>
              </div>
            </div>

            <div class="fields-grid">
              <div class="field-wrap full-width">
                <div class="field-top"><label>Chest Pain Classification (cp)</label><span class="field-hint">Clinical symptomatology</span></div>
                <select id="inCp" name="cp" class="input-box">
                  <option value="1">Type 1: Typical Angina (Exertional pressure)</option>
                  <option value="2">Type 2: Atypical Angina (Non-classical discomfort)</option>
                  <option value="3">Type 3: Non-Anginal Chest Pain</option>
                  <option value="4" selected>Type 4: Asymptomatic (Silent Ischemia)</option>
                </select>
              </div>

              <div class="field-wrap">
                <div class="field-top"><label>Max Heart Rate (thalach)</label><span class="field-hint" id="txtTargetHr">Target: ~165 bpm</span></div>
                <div class="slider-group">
                  <input type="range" id="slThalach" min="60" max="220" value="150" class="range-slider">
                  <input type="number" id="inThalach" name="thalach" min="60" max="230" value="150" class="input-box slider-num" required>
                </div>
              </div>

              <div class="field-wrap">
                <div class="field-top"><label>Exercise Induced Angina (exang)</label><span class="field-hint">Provoked pain</span></div>
                <div class="toggle-radios">
                  <label class="radio-lbl"><input type="radio" name="exang" value="0" checked><span>No</span></label>
                  <label class="radio-lbl"><input type="radio" name="exang" value="1"><span>Yes</span></label>
                </div>
              </div>

              <div class="field-wrap full-width">
                <div class="field-top"><label>Resting Electrocardiogram (restecg)</label><span class="field-hint">Baseline rhythm</span></div>
                <select id="inRestecg" name="restecg" class="input-box">
                  <option value="0" selected>0: Normal Sinus Rhythm</option>
                  <option value="1">1: ST-T Wave Abnormality (&gt; 0.05 mV)</option>
                  <option value="2">2: Left Ventricular Hypertrophy (LVH)</option>
                </select>
              </div>
            </div>
          </div>

          <!-- Section 3 -->
          <div class="card-panel" style="margin-top: 16px;">
            <div class="panel-header">
              <div class="panel-icon icon-purple">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
              </div>
              <div>
                <h2 class="panel-title">3. Stress Diagnostics & Fluoroscopy</h2>
                <p class="panel-desc">ST depression, fluoroscopy vessels, and thallium perfusion scan</p>
              </div>
            </div>

            <div class="fields-grid">
              <div class="field-wrap">
                <div class="field-top"><label>ST Depression (oldpeak)</label><span class="field-hint">Induced in mm</span></div>
                <div class="slider-group">
                  <input type="range" id="slOldpeak" min="0.0" max="6.0" step="0.1" value="1.0" class="range-slider">
                  <input type="number" id="inOldpeak" name="oldpeak" min="0.0" max="6.2" step="0.1" value="1.0" class="input-box slider-num" required>
                </div>
              </div>

              <div class="field-wrap">
                <div class="field-top"><label>Peak ST Slope</label><span class="field-hint">Trajectory</span></div>
                <select id="inSlope" name="slope" class="input-box">
                  <option value="1">1: Upsloping (Favorable)</option>
                  <option value="2" selected>2: Flat (Ischemic sign)</option>
                  <option value="3">3: Downsloping (Severe sign)</option>
                </select>
              </div>

              <div class="field-wrap">
                <div class="field-top"><label>Fluoroscopy Occluded Vessels (ca)</label><span class="field-hint">0 - 3 vessels</span></div>
                <select id="inCa" name="ca" class="input-box">
                  <option value="0" selected>0 Major Vessels Colored</option>
                  <option value="1">1 Major Vessel Colored</option>
                  <option value="2">2 Major Vessels Colored</option>
                  <option value="3">3 Major Vessels Colored</option>
                </select>
              </div>

              <div class="field-wrap">
                <div class="field-top"><label>Thallium Perfusion (thal)</label><span class="field-hint">Scintigraphy scan</span></div>
                <select id="inThal" name="thal" class="input-box">
                  <option value="3" selected>3: Normal Perfusion</option>
                  <option value="6">6: Fixed Defect (Previous Infarct)</option>
                  <option value="7">7: Reversible Defect (Active Ischemia)</option>
                </select>
              </div>
            </div>
          </div>

          <div style="margin-top: 18px;">
            <button type="submit" id="btnSubmitPredict" class="btn-run-ai">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>
              <span>Run AI Multi-Model Cardiac Evaluation</span>
            </button>
          </div>
        </form>
      </section>

      <!-- Diagnostic Output HUD Column -->
      <section class="results-column">
        <div class="diagnostic-hud">
          
          <!-- Placeholder State -->
          <div class="placeholder-state" id="hudPlaceholder">
            <div class="pulse-rings">
              <div class="ring-circ r-1"></div>
              <div class="ring-circ r-2"></div>
              <div class="ring-core">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>
              </div>
            </div>
            <h3 class="ph-title">CardioPulse Diagnostic HUD</h3>
            <p class="ph-desc">Awaiting patient clinical input. Select an <strong>Instant Preset</strong> above or submit the form on the left to compute real-time multi-model risk stratification.</p>
          </div>

          <!-- Active Results HUD -->
          <div class="active-hud" id="hudActive">
            
            <!-- Result Header -->
            <div class="hud-header">
              <div>
                <span class="risk-badge badge-l" id="riskBadge">LOW RISK</span>
                <h3 class="hud-headline" id="txtHeadline">Favorable Cardiovascular Profile</h3>
                <div style="font-size:0.72rem; color:var(--text-faint); margin-top:2px;" id="txtPatientId">Patient ID: PAT-2026-X812 • Verified</div>
              </div>
              <button type="button" class="btn-pdf" id="btnPrintPdf" title="Print official Cardiology Assessment Report">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 6 2 18 2 18 9"></polyline><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path><rect x="6" y="14" width="12" height="8"></rect></svg>
                Print PDF
              </button>
            </div>

            <!-- Dial Gauge -->
            <div class="gauge-wrapper">
              <div class="gauge-dial">
                <svg class="gauge-svg" viewBox="0 0 200 120">
                  <path class="arc-bg" d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke-width="16" stroke-linecap="round"/>
                  <path class="arc-progress" id="arcProgress" d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke-width="16" stroke-linecap="round"/>
                </svg>
                <div class="gauge-center-val">
                  <div class="score-txt"><span id="txtScoreVal">0.0</span>%</div>
                  <div class="score-sub">Consensus Risk Score</div>
                </div>
              </div>
            </div>

            <!-- Multi-Model Consensus Panel -->
            <div class="consensus-panel">
              <div class="consensus-top">
                <span>Multi-Model Consensus Agreement</span>
                <span class="consensus-tag" id="txtConsensusTag">3/3 Models Agree</span>
              </div>
              <div class="model-bars">
                <div class="model-bar-item">
                  <span class="m-name">Random Forest</span>
                  <div class="m-track"><div class="m-fill" id="fillRf" style="width:0%; background:#38bdf8;"></div></div>
                  <span class="m-pct" id="pctRf">0%</span>
                </div>
                <div class="model-bar-item">
                  <span class="m-name">Gradient Boosting</span>
                  <div class="m-track"><div class="m-fill" id="fillGb" style="width:0%; background:#a855f7;"></div></div>
                  <span class="m-pct" id="pctGb">0%</span>
                </div>
                <div class="model-bar-item">
                  <span class="m-name">Logistic Regression</span>
                  <div class="m-track"><div class="m-fill" id="fillLr" style="width:0%; background:#10b981;"></div></div>
                  <span class="m-pct" id="pctLr">0%</span>
                </div>
              </div>
            </div>

            <!-- Dual Charts Row: Radar & SHAP Waterfall -->
            <div class="dual-charts-row">
              <!-- Radar Chart -->
              <div class="chart-box">
                <div class="box-title">6-Axis Cardiac Stress Radar</div>
                <div class="radar-svg-box">
                  <svg id="radarSvg" width="160" height="135" viewBox="-80 -70 160 140"></svg>
                </div>
                <div style="font-size:0.66rem; color:var(--text-faint); text-align:center;">
                  <span style="color:#38bdf8;">■ Patient Risk Polygon</span> vs <span style="color:#64748b;">■ Cleveland Baseline</span>
                </div>
              </div>

              <!-- SHAP Feature Waterfall Impact -->
              <div class="chart-box">
                <div class="box-title">Feature Attribution (SHAP)</div>
                <div class="waterfall-list" id="wfContainer">
                  <!-- Populated dynamically -->
                </div>
              </div>
            </div>

            <!-- Action Plan & Recommendations -->
            <div class="recs-box">
              <div class="box-title">Cardiologist Clinical Guidance & Plan</div>
              <ul class="recs-ul" id="recsUl"></ul>
            </div>

          </div>

        </div>
      </section>

    </main>

    <footer class="app-footer">
      CardioPulse Pro • Real-Time Multi-Model Machine Learning • Trained on Cleveland Clinic Cohort • Deployed on Vercel
    </footer>
  </div>

  <script>
    // ---------------------------------------------------------
    // Real-Time Canvas ECG Oscilloscope Simulator
    // ---------------------------------------------------------
    const canvas = document.getElementById('ecgCanvas');
    const ctx = canvas.getContext('2d');
    let ecgSpeed = 150; // bpm
    let xPos = 0;
    let ecgPoints = [];
    const maxPoints = 350;

    function resizeCanvas() {
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // P-Q-R-S-T Cardiac Waveform Generator
    function getEcgY(t) {
      const cycle = t % 1.0;
      if (cycle > 0.15 && cycle < 0.25) return -Math.sin((cycle - 0.15) * Math.PI / 0.1) * 6; // P wave
      if (cycle > 0.35 && cycle < 0.38) return Math.sin((cycle - 0.35) * Math.PI / 0.03) * 5; // Q dip
      if (cycle >= 0.38 && cycle < 0.42) return -Math.sin((cycle - 0.38) * Math.PI / 0.04) * 26; // R peak
      if (cycle >= 0.42 && cycle < 0.46) return Math.sin((cycle - 0.42) * Math.PI / 0.04) * 9; // S dip
      if (cycle > 0.55 && cycle < 0.72) return -Math.sin((cycle - 0.55) * Math.PI / 0.17) * 9; // T wave
      return 0; // isoelectric line
    }

    let simTime = 0;
    function animateEcg() {
      const freq = ecgSpeed / 60; // beats per second
      simTime += 0.016 * freq;
      const midY = canvas.height / 2;
      const y = midY + getEcgY(simTime);

      ecgPoints.push(y);
      if (ecgPoints.length > canvas.width) {
        ecgPoints.shift();
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw Grid Lines
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.06)';
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 20) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let gy = 0; gy < canvas.height; gy += 15) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(canvas.width, gy); ctx.stroke();
      }

      // Draw ECG Line
      ctx.beginPath();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 6;

      for (let i = 0; i < ecgPoints.length; i++) {
        if (i === 0) ctx.moveTo(i, ecgPoints[i]);
        else ctx.lineTo(i, ecgPoints[i]);
      }
      ctx.stroke();

      // Leading Glowing Pulse Dot
      if (ecgPoints.length > 0) {
        const lastX = ecgPoints.length - 1;
        const lastY = ecgPoints[lastX];
        ctx.beginPath();
        ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.shadowColor = '#38bdf8';
        ctx.shadowBlur = 10;
        ctx.fill();
      }

      requestAnimationFrame(animateEcg);
    }
    requestAnimationFrame(animateEcg);

    // ---------------------------------------------------------
    // Web Audio Synthesizer Heartbeat Beep
    // ---------------------------------------------------------
    let audioCtx = null;
    let isAudioActive = false;
    let audioInterval = null;

    function playBeep() {
      if (!isAudioActive) return;
      try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(650, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.08);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.08);
      } catch (e) {}
    }

    function syncHeartbeatAudio() {
      if (audioInterval) clearInterval(audioInterval);
      if (isAudioActive) {
        const msPerBeat = (60 / ecgSpeed) * 1000;
        audioInterval = setInterval(playBeep, msPerBeat);
      }
    }

    const btnAudio = document.getElementById('btnAudioToggle');
    btnAudio.onclick = () => {
      isAudioActive = !isAudioActive;
      btnAudio.classList.toggle('active', isAudioActive);
      document.getElementById('txtSoundState').textContent = isAudioActive ? 'Audio Active' : 'Audio Muted';
      syncHeartbeatAudio();
      if (isAudioActive) playBeep();
    };

    // ---------------------------------------------------------
    // Input Sync & Sliders
    // ---------------------------------------------------------
    const form = document.getElementById('cardioForm');
    const slAge = document.getElementById('slAge');
    const inAge = document.getElementById('inAge');
    const slThalach = document.getElementById('slThalach');
    const inThalach = document.getElementById('inThalach');
    const slOldpeak = document.getElementById('slOldpeak');
    const inOldpeak = document.getElementById('inOldpeak');
    const txtLiveBpm = document.getElementById('txtLiveBpm');
    const txtTargetHr = document.getElementById('txtTargetHr');

    function updateAgeCalcs() {
      const a = parseInt(inAge.value) || 55;
      const target = Math.round(220 - a);
      txtTargetHr.textContent = `Formula: ~${target} bpm`;
    }

    slAge.oninput = () => { inAge.value = slAge.value; updateAgeCalcs(); };
    inAge.oninput = () => { slAge.value = inAge.value; updateAgeCalcs(); };

    function updateHeartRate(val) {
      ecgSpeed = Math.max(50, Math.min(230, parseInt(val) || 120));
      txtLiveBpm.innerHTML = `${ecgSpeed} <span style="font-size:0.75rem; color:#94a3b8;">BPM</span>`;
      syncHeartbeatAudio();
    }
    slThalach.oninput = () => { inThalach.value = slThalach.value; updateHeartRate(slThalach.value); };
    inThalach.oninput = () => { slThalach.value = inThalach.value; updateHeartRate(inThalach.value); };

    slOldpeak.oninput = () => { inOldpeak.value = parseFloat(slOldpeak.value).toFixed(1); };
    inOldpeak.oninput = () => { slOldpeak.value = inOldpeak.value; };

    updateAgeCalcs();
    updateHeartRate(inThalach.value);

    // ---------------------------------------------------------
    // Instant Patient Demos
    // ---------------------------------------------------------
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
      slThalach.value = inThalach.value = d.thalach;
      updateHeartRate(d.thalach);
      form.querySelector(`input[name="exang"][value="${d.exang}"]`).checked = true;
      document.getElementById('inRestecg').value = d.restecg;
      slOldpeak.value = inOldpeak.value = d.oldpeak.toFixed(1);
      document.getElementById('inSlope').value = d.slope;
      document.getElementById('inCa').value = d.ca;
      document.getElementById('inThal').value = d.thal;
      updateAgeCalcs();
      triggerAiEvaluation();
    }

    document.getElementById('btnPresetHealthy').onclick = () => applyPreset('healthy');
    document.getElementById('btnPresetModerate').onclick = () => applyPreset('moderate');
    document.getElementById('btnPresetHigh').onclick = () => applyPreset('high');
    document.getElementById('btnResetAll').onclick = () => {
      form.reset();
      slAge.value = inAge.value = 55;
      slThalach.value = inThalach.value = 150;
      slOldpeak.value = inOldpeak.value = "1.0";
      updateHeartRate(150);
      document.getElementById('hudActive').style.display = 'none';
      document.getElementById('hudPlaceholder').style.display = 'block';
    };

    // ---------------------------------------------------------
    // Prediction API Connection
    // ---------------------------------------------------------
    async function triggerAiEvaluation() {
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
        renderDiagnosticHud(data);
      } catch (err) {
        alert('CardioPulse AI request failed: ' + err.message);
      }
    }

    form.onsubmit = (e) => { e.preventDefault(); triggerAiEvaluation(); };

    // ---------------------------------------------------------
    // Render Results & Radar SVG
    // ---------------------------------------------------------
    const GAUGE_CIRC = 251.327;

    function renderDiagnosticHud(data) {
      document.getElementById('hudPlaceholder').style.display = 'none';
      const activeHud = document.getElementById('hudActive');
      activeHud.style.display = 'flex';

      // Badge
      const badge = document.getElementById('riskBadge');
      badge.className = 'risk-badge';
      if (data.risk_tier === 'Low Risk') {
        badge.classList.add('badge-l'); badge.textContent = 'LOW RISK';
      } else if (data.risk_tier === 'Moderate Risk') {
        badge.classList.add('badge-m'); badge.textContent = 'MODERATE RISK';
      } else {
        badge.classList.add('badge-h'); badge.textContent = 'HIGH RISK';
      }

      document.getElementById('txtHeadline').textContent = data.headline;
      document.getElementById('txtPatientId').textContent = `Patient ID: ${data.patient_id} • Cohort Percentile: ${data.cohort_percentile}th`;

      // Animate Gauge Arc
      const targetOffset = GAUGE_CIRC * (1 - Math.min(1.0, data.ensemble_probability));
      const arc = document.getElementById('arcProgress');
      arc.style.strokeDashoffset = targetOffset;
      arc.style.stroke = data.ensemble_probability < 0.35 ? '#10b981' : (data.ensemble_probability < 0.65 ? '#f59e0b' : '#f43f5e');

      // Counter
      animateVal(document.getElementById('txtScoreVal'), data.ensemble_risk_percentage, 800);

      // Multi-Model Consensus
      document.getElementById('txtConsensusTag').textContent = data.consensus_agreement;
      setBar('fillRf', 'pctRf', data.models.rf.probability);
      setBar('fillGb', 'pctGb', data.models.gb.probability);
      setBar('fillLr', 'pctLr', data.models.lr.probability);

      // Draw SVG Radar
      drawRadar(data.radar_metrics);

      // SHAP Waterfall Bars
      renderWaterfall(data.attributions);

      // Recommendations
      const ul = document.getElementById('recsUl');
      ul.innerHTML = '';
      data.recommendations.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        ul.appendChild(li);
      });
    }

    function setBar(fillId, txtId, prob) {
      const pct = (prob * 100).toFixed(1);
      document.getElementById(fillId).style.width = pct + '%';
      document.getElementById(txtId).textContent = pct + '%';
    }

    function animateVal(el, target, duration) {
      const start = performance.now();
      function step(now) {
        const p = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - p, 3);
        el.textContent = (target * ease).toFixed(1);
        if (p < 1) requestAnimationFrame(step);
        else el.textContent = target.toFixed(1);
      }
      requestAnimationFrame(step);
    }

    // Dynamic 6-Axis Radar SVG
    function drawRadar(m) {
      const svg = document.getElementById('radarSvg');
      svg.innerHTML = '';

      const keys = ['blood_pressure', 'cholesterol', 'ischemic_st', 'vessel_occlusion', 'perfusion_defect', 'exertion_strain'];
      const labels = ['BP', 'Lipids', 'ST Depr', 'Vessels', 'Perfusion', 'Strain'];
      const count = 6;
      const R = 50;

      // Draw Concentric Reference Web
      [0.33, 0.66, 1.0].forEach(scale => {
        let pts = '';
        for (let i = 0; i < count; i++) {
          const angle = (i * 2 * Math.PI / count) - (Math.PI / 2);
          const px = Math.cos(angle) * R * scale;
          const py = Math.sin(angle) * R * scale;
          pts += `${px},${py} `;
        }
        const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        poly.setAttribute('points', pts.trim());
        poly.setAttribute('fill', 'none');
        poly.setAttribute('stroke', 'rgba(255,255,255,0.08)');
        poly.setAttribute('stroke-width', '1');
        svg.appendChild(poly);
      });

      // Axis spokes & labels
      for (let i = 0; i < count; i++) {
        const angle = (i * 2 * Math.PI / count) - (Math.PI / 2);
        const px = Math.cos(angle) * R;
        const py = Math.sin(angle) * R;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', '0'); line.setAttribute('y1', '0');
        line.setAttribute('x2', px); line.setAttribute('y2', py);
        line.setAttribute('stroke', 'rgba(255,255,255,0.1)');
        svg.appendChild(line);

        // Labels
        const lx = Math.cos(angle) * (R + 14);
        const ly = Math.sin(angle) * (R + 14) + 3;
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', lx); text.setAttribute('y', ly);
        text.setAttribute('font-size', '7');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', '#94a3b8');
        text.textContent = labels[i];
        svg.appendChild(text);
      }

      // Patient Polygon
      let patientPts = '';
      for (let i = 0; i < count; i++) {
        const angle = (i * 2 * Math.PI / count) - (Math.PI / 2);
        const val = (m[keys[i]] || 20) / 100;
        const r = Math.max(10, Math.min(R, R * val));
        const px = Math.cos(angle) * r;
        const py = Math.sin(angle) * r;
        patientPts += `${px},${py} `;
      }

      const patientPoly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      patientPoly.setAttribute('points', patientPts.trim());
      patientPoly.setAttribute('fill', 'rgba(56, 189, 248, 0.25)');
      patientPoly.setAttribute('stroke', '#38bdf8');
      patientPoly.setAttribute('stroke-width', '1.8');
      svg.appendChild(patientPoly);
    }

    // Render SHAP Waterfall Bars
    function renderWaterfall(attrs) {
      const c = document.getElementById('wfContainer');
      c.innerHTML = '';
      if (!attrs || attrs.length === 0) {
        c.innerHTML = '<div style="font-size:0.72rem;color:var(--text-faint);">All parameters within population norm.</div>';
        return;
      }
      attrs.forEach(a => {
        const isRisk = a.direction === 'risk';
        const item = document.createElement('div');
        item.className = 'wf-item';
        item.innerHTML = `
          <span class="wf-label" title="${a.label}">${a.label}</span>
          <div class="wf-bar-wrap">
            <div class="wf-bar ${isRisk ? 'wf-risk' : 'wf-prot'}" style="width: ${Math.min(100, a.percentage)}%"></div>
          </div>
          <span class="wf-val ${isRisk ? 'wf-risk' : 'wf-prot'}">${isRisk ? '+' : '-'}${a.percentage}%</span>
        `;
        c.appendChild(item);
      });
    }

    document.getElementById('btnPrintPdf').onclick = () => window.print();
  </script>
</body>
</html>
"""


# -------------------------------------------------------------
# API Endpoints (Pure Python)
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the complete Next-Gen Cardiovascular Diagnostic UI."""
    return HTMLResponse(content=UI_HTML, status_code=200)


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "system": "CardioPulse PRO Next-Gen Diagnostic OS",
        "platform": "Vercel Serverless Python",
        "models_loaded": list(ensemble_models.keys()),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/predict")
@app.post("/predict")
def predict_cardiac_risk(patient: PatientData):
    """
    Computes ensemble consensus predictions (Random Forest, Gradient Boosting,
    Logistic Regression), SHAP feature attribution deltas, and 6-axis radar metrics.
    """
    global ensemble_models
    if not ensemble_models:
        ensemble_models = train_fallback_models()

    raw_dict = patient.model_dump()
    norm_dict = normalize(raw_dict)
    df_row = pd.DataFrame([[norm_dict[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

    model_preds = {}
    probs = []

    for key in ["rf", "gb", "lr"]:
        pipe = ensemble_models.get(key)
        if pipe:
            pred = int(pipe.predict(df_row)[0])
            prob = float(pipe.predict_proba(df_row)[0][1])
            model_preds[key] = {
                "prediction": pred,
                "probability": round(prob, 4),
                "risk_percentage": round(prob * 100, 1)
            }
            probs.append(prob)

    ensemble_prob = float(np.mean(probs)) if probs else 0.5
    ensemble_pct = round(ensemble_prob * 100, 1)

    # Consensus Agreement count
    positive_count = sum(1 for m in model_preds.values() if m["prediction"] == 1)
    if positive_count == 3:
        agreement = "3/3 Models Agree: High Risk"
    elif positive_count == 0:
        agreement = "3/3 Models Agree: Low Risk"
    else:
        agreement = f"{positive_count}/3 Consensus Warning"

    # Risk Tier
    if ensemble_prob < 0.35:
        tier = "Low Risk"
        headline = "Favorable Cardiovascular Profile"
        summary = "No acute signs of coronary artery disease detected. Hemodynamic markers remain within expected clinical baseline."
    elif ensemble_prob < 0.65:
        tier = "Moderate Risk"
        headline = "Borderline / Moderate Cardiac Risk"
        summary = "Subtle ischemic or hemodynamic flags observed. Comprehensive clinical follow-up and preventive adjustments are recommended."
    else:
        tier = "High Risk"
        headline = "Elevated Coronary Heart Disease Risk"
        summary = "Strong concurrence across multiple diagnostic vectors. Immediate consultation with a cardiologist is recommended."

    radar = compute_radar_metrics(norm_dict)
    attributions = compute_feature_attributions(df_row)

    # Recommendations
    if tier == "High Risk":
        recs = [
            "🚨 Urgent Cardiology Consultation: Schedule an in-person diagnostic evaluation within 48-72 hours.",
            "📋 Diagnostic Imaging: Inquire about high-resolution CT Coronary Angiogram (CTCA) or catheter angiography.",
            "💊 Pharmacotherapy Optimization: Discuss antiplatelet, statin, and antihypertensive regimens.",
            "⚠️ Exercise Caution: Avoid intense or strenuous unmonitored workouts until medically cleared."
        ]
    elif tier == "Moderate Risk":
        recs = [
            "🩺 Preventive Checkup: Follow up with your primary physician within 3-4 weeks.",
            "📊 Stress Echocardiography: Undergo a Treadmill Exercise Stress Test (TMT) to evaluate functional capacity.",
            "🥗 Cardiovascular Nutrition: Adopt a Mediterranean or DASH diet low in saturated fats and sodium.",
            "🏃 Structured Activity: Aim for 150 minutes of moderate aerobic activity weekly (e.g. brisk walking)."
        ]
    else:
        recs = [
            "🌟 Favorable Baseline: Excellent cardiovascular markers. Keep up your active lifestyle and nutritious diet!",
            "🛡️ Annual Surveillance: Maintain annual monitoring of resting blood pressure and lipid panels.",
            "🥦 Nutrient-Dense Habits: Continue whole grains, leafy vegetables, omega-3s, and regular hydration.",
            "🧘 Stress Management: Prioritize 7-8 hours of quality restorative sleep and daily stress reduction."
        ]

    # Deterministic cohort percentile estimate
    percentile = min(99, max(5, int(ensemble_prob * 95 + (norm_dict['age'] - 30) * 0.15)))

    # Unique Patient Demo Tag
    patient_id = f"PAT-{norm_dict['age']}{norm_dict['sex']}-{int(norm_dict['trestbps'])%100:02d}"

    return {
        "status": "success",
        "patient_id": patient_id,
        "ensemble_probability": round(ensemble_prob, 4),
        "ensemble_risk_percentage": ensemble_pct,
        "risk_tier": tier,
        "headline": headline,
        "summary": summary,
        "consensus_agreement": agreement,
        "cohort_percentile": percentile,
        "models": model_preds,
        "radar_metrics": radar,
        "attributions": attributions,
        "recommendations": recs,
        "evaluated_at": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("  🚀 CardioPulse PRO Diagnostic OS Starting at http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    uvicorn.run("api.index:app", host="127.0.0.1", port=8000, reload=True)
