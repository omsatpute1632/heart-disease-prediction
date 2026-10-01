"""
Heart Disease Prediction - Model Training Script
Trains and evaluates multiple Machine Learning classifiers on heart.csv
Exports the best scikit-learn Pipeline and performance metrics.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_OUTPUT_PATH = os.path.join(MODEL_DIR, "heart_model.joblib")
METRICS_OUTPUT_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Features list matching UCI Cleveland heart.csv standard
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]
TARGET_COL = "target"


def load_and_preprocess_data(csv_path: str):
    """Loads and validates the heart dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    logging.info(f"Loaded dataset from {csv_path} with shape {df.shape}")

    # Standardize column names (lowercase, strip whitespace)
    df.columns = [col.strip().lower() for col in df.columns]

    # Verify all expected columns are present
    missing_cols = [col for col in FEATURE_NAMES + [TARGET_COL] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")

    # Check and handle missing/null values if any
    null_counts = df[FEATURE_NAMES + [TARGET_COL]].isnull().sum()
    if null_counts.sum() > 0:
        logging.warning(f"Found null values, imputing with median:\n{null_counts[null_counts > 0]}")
        df = df.fillna(df.median(numeric_only=True))

    X = df[FEATURE_NAMES]
    y = df[TARGET_COL]

    logging.info(f"Class distribution - Target 0 (No Disease): {(y == 0).sum()}, Target 1 (Disease): {(y == 1).sum()}")
    return X, y


def train_and_evaluate():
    """Trains candidate models, selects the best, and exports pipeline + metrics."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_and_preprocess_data(DATA_PATH)

    # Train-test split (80-20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Candidate models wrapped in standard scaling pipelines
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_split=4,
                random_state=42
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=3,
                random_state=42
            ))
        ])
    }

    best_name = None
    best_pipeline = None
    best_score = -1.0
    results = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, pipeline in models.items():
        # Cross validation score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        mean_cv_auc = float(np.mean(cv_scores))

        # Train on full train split
        pipeline.fit(X_train, y_train)

        # Predictions on holdout test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        roc_auc = float(roc_auc_score(y_test, y_proba))

        logging.info(
            f"[{name}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | CV AUC: {mean_cv_auc:.4f}"
        )

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "cv_auc_mean": round(mean_cv_auc, 4)
        }

        # Select model based on combined ROC-AUC and F1
        combined_score = (roc_auc + f1) / 2.0
        if combined_score > best_score:
            best_score = combined_score
            best_name = name
            best_pipeline = pipeline

    logging.info(f"🏆 Best Model Selected: {best_name}")

    # Retrain best pipeline on full dataset (X, y) for maximum production predictive power
    best_pipeline.fit(X, y)

    # Extract feature importances if available
    feature_importances = {}
    clf = best_pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        for feat, imp in zip(FEATURE_NAMES, clf.feature_importances_):
            feature_importances[feat] = round(float(imp), 4)
    elif hasattr(clf, "coef_"):
        for feat, coef in zip(FEATURE_NAMES, clf.coef_[0]):
            feature_importances[feat] = round(float(abs(coef)), 4)

    # Sort feature importances descending
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    # Save best model pipeline
    joblib.dump(best_pipeline, MODEL_OUTPUT_PATH)
    logging.info(f"Saved trained pipeline to: {MODEL_OUTPUT_PATH}")

    # Save metrics and metadata
    metrics_data = {
        "best_model": best_name,
        "trained_at": datetime.now().isoformat(),
        "dataset_rows": len(X),
        "features": FEATURE_NAMES,
        "evaluation_metrics": results[best_name],
        "all_model_benchmarks": results,
        "feature_importances": feature_importances
    }

    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logging.info(f"Saved evaluation metrics to: {METRICS_OUTPUT_PATH}")

    print("\n" + "=" * 50)
    print("      HEART DISEASE MODEL TRAINING COMPLETE       ")
    print("=" * 50)
    print(f"Selected Model:   {best_name}")
    print(f"Holdout Accuracy: {results[best_name]['accuracy'] * 100:.2f}%")
    print(f"Holdout ROC-AUC:  {results[best_name]['roc_auc'] * 100:.2f}%")
    print(f"Holdout F1 Score: {results[best_name]['f1_score'] * 100:.2f}%")
    print(f"Model Artifact:   {MODEL_OUTPUT_PATH}")
    print(f"Metrics File:     {METRICS_OUTPUT_PATH}")
    print("Top Risk Predictors:", list(feature_importances.keys())[:5])
    print("=" * 50 + "\n")


if __name__ == "__main__":
    train_and_evaluate()
