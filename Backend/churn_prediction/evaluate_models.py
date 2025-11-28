import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np

import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Paths
BASE_DIR = os.path.dirname(__file__)
rfm_scaled_path = os.path.join(BASE_DIR, '..', 'src/static', 'rfm_scaled_result.csv')
rfm_path = os.path.join(BASE_DIR, '..', 'src/static', 'rfm_result.csv')

# Load data
rfm_scaled = pd.read_csv(rfm_scaled_path)
rfm = pd.read_csv(rfm_path)

# Labels (same business rule as before)
rfm['Churn'] = (rfm['Recency'] > 60).astype(int)
X = rfm_scaled
y = rfm['Churn']

# Dictionary to hold results
results = {}

def evaluate(y_true, y_pred, y_prob):
    """Compute metrics for evaluation"""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_prob),
    }

# --- ANN ---
ann_path = os.path.join(BASE_DIR, "models", "ann", "churn_model.h5")
if os.path.exists(ann_path):
    ann_model = tf.keras.models.load_model(ann_path)
    y_prob = ann_model.predict(X).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    results["ANN"] = evaluate(y, y_pred, y_prob)

# --- Logistic Regression ---
logreg_path = os.path.join(BASE_DIR, "models", "logreg", "churn_model_logreg.pkl")
if os.path.exists(logreg_path):
    logreg_model = joblib.load(logreg_path)
    y_prob = logreg_model.predict_proba(X)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    results["Logistic Regression"] = evaluate(y, y_pred, y_prob)

# --- Random Forest ---
rf_path = os.path.join(BASE_DIR, "models", "rf", "churn_model_rf.pkl")
if os.path.exists(rf_path):
    rf_model = joblib.load(rf_path)
    y_prob = rf_model.predict_proba(X)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    results["Random Forest"] = evaluate(y, y_pred, y_prob)

# --- XGBoost ---
xgb_path = os.path.join(BASE_DIR, "models", "xgb", "churn_model_xgb.pkl")
if os.path.exists(xgb_path):
    xgb_model = joblib.load(xgb_path)
    y_prob = xgb_model.predict_proba(X)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    results["XGBoost"] = evaluate(y, y_pred, y_prob)

# ---- Display Results ----
df_results = pd.DataFrame(results).T.round(4)  # transpose for readability
print("\n📊 Model Comparison:\n")
print(df_results)
