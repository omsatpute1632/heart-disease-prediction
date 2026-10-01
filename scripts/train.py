"""
CardioPulse AI — Advanced Multi-Model Training Script
Trains Random Forest, Gradient Boosting, and Logistic Regression on heart.csv
Exports ensemble pipelines, benchmarks, and feature weights.
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble_models.joblib")
SINGLE_MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def train_advanced_ensemble():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42))
        ]),
        "gb": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42))
        ]),
        "lr": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ])
    }

    metrics = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for key, pipe in models.items():
        cv_auc = float(np.mean(cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")))
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics[key] = {
            "name": "Random Forest" if key == "rf" else ("Gradient Boosting" if key == "gb" else "Logistic Regression"),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
            "cv_auc": round(cv_auc, 4)
        }

    # Retrain on full dataset
    for key, pipe in models.items():
        pipe.fit(X, y)

    # Save ensemble & primary model
    joblib.dump(models, ENSEMBLE_PATH)
    joblib.dump(models["rf"], SINGLE_MODEL_PATH)

    # Feature weights from Random Forest
    rf_clf = models["rf"].named_steps["clf"]
    importances = {f: round(float(imp), 4) for f, imp in zip(FEATURE_NAMES, rf_clf.feature_importances_)}
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    # Population baseline means & standard deviations
    pop_stats = {
        "means": {f: round(float(X[f].mean()), 2) for f in FEATURE_NAMES},
        "stds": {f: round(float(X[f].std()), 2) for f in FEATURE_NAMES}
    }

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "dataset_rows": len(df),
        "features": FEATURE_NAMES,
        "models": metrics,
        "primary_model": "Random Forest Classifier",
        "feature_importances": importances,
        "population_baseline": pop_stats
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 55)
    print("      MULTI-MODEL ENSEMBLE TRAINING COMPLETE       ")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"[{v['name']:<20}] Accuracy: {v['accuracy']*100:.1f}% | ROC-AUC: {v['roc_auc']*100:.1f}% | F1: {v['f1_score']*100:.1f}%")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    train_advanced_ensemble()
