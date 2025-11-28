import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
from flask import Flask, jsonify
import joblib
import tensorflow as tf

from config import MODEL_TYPE

app = Flask(__name__)

# ---- Load the right model based on config ----
if MODEL_TYPE == "ann":
    model = tf.keras.models.load_model(
        os.path.join(os.path.dirname(__file__), "models", "ann", "churn_model.h5")
    )
elif MODEL_TYPE == "logreg":
    model = joblib.load(
        os.path.join(os.path.dirname(__file__), "models", "logreg", "churn_model_logreg.pkl")
    )
elif MODEL_TYPE == "rf":
    model = joblib.load(
        os.path.join(os.path.dirname(__file__), "models", "rf", "churn_model_rf.pkl")
    )
elif MODEL_TYPE == "xgb":
    model = joblib.load(
        os.path.join(os.path.dirname(__file__), "models", "xgb", "churn_model_xgb.pkl")
    )
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

# ---- Prediction Route ----
@app.route('/predict', methods=['POST'])
def predict():
    # Load datasets
    rfm_scaled_path = os.path.join(os.path.dirname(__file__), '..', 'src/static', 'rfm_scaled_result.csv')
    rfm_scaled = pd.read_csv(rfm_scaled_path)
    rfm_path = os.path.join(os.path.dirname(__file__), '..', 'src/static', 'rfm_result.csv')
    rfm = pd.read_csv(rfm_path)

    # Predictions depend on model type
    if MODEL_TYPE == "ann":
        churn_probs = model.predict(rfm_scaled).flatten()
        churn_labels = (churn_probs > 0.5).astype(int)
    else:
        churn_probs = model.predict_proba(rfm_scaled)[:, 1]
        churn_labels = (churn_probs > 0.5).astype(int)

    # Add predictions to DataFrame
    rfm["Churn_Label"] = churn_labels
    rfm["Churn_Probability"] = churn_probs

    # Save CSV results
    output_path = os.path.join(os.path.dirname(__file__), "static", f"churn_results_{MODEL_TYPE}.csv")
    rfm.to_csv(output_path, index=False)

    # Return JSON
    return jsonify(rfm.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)
