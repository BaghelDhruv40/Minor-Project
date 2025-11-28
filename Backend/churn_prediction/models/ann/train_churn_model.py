
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'src')))

from utils import preprocess_data
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Path to the dataset
rfm_scaled_path = os.path.join(os.path.dirname(__file__), '../../..', 'src/static', 'rfm_scaled_result.csv')
rfm_path = os.path.join(os.path.dirname(__file__), '../../..', 'src/static', 'rfm_result.csv')


# Preprocess data
# rfm, rfm_scaled = preprocess_data(data_path)

rfm_scaled=pd.read_csv(rfm_scaled_path)
rfm=pd.read_csv(rfm_path)

# Labeling using business rule
rfm['Churn'] = (rfm['Recency'] > 60).astype(int)
# rfm['Churn'] = (
#     (rfm['Recency'] > 60) &   
#     (rfm['Frequency'] < 10) &  
#     (rfm['Amount'] < 1000)     
# ).astype(int)


X = rfm_scaled
y = rfm['Churn']

# Build ANN
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# Save model to churn_prediction folder
save_path = os.path.join(os.path.dirname(__file__), 'churn_model.h5')
model.save(save_path)
# print(f"Model saved to {save_path}")
