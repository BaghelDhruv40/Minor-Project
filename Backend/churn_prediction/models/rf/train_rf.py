# train_churn_model_logreg.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'src')))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Paths
rfm_scaled_path = os.path.join(os.path.dirname(__file__), '../../..', 'src/static', 'rfm_scaled_result.csv')
rfm_path = os.path.join(os.path.dirname(__file__), '../../..', 'src/static', 'rfm_result.csv')

# Load data
rfm_scaled = pd.read_csv(rfm_scaled_path)
rfm = pd.read_csv(rfm_path)

# Label
rfm['Churn'] = (rfm['Recency'] > 60).astype(int)

X = rfm_scaled
y = rfm['Churn']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

save_path = os.path.join(os.path.dirname(__file__), 'churn_model_rf.pkl')
joblib.dump(model, save_path)
