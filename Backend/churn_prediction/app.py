import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd

from utils import preprocess_data
import tensorflow as tf
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load ANN churn model
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'churn_model.h5'))

@app.route('/') # Ignore this route for now. Instead of a separate route, button based trigger will be implemented from the main home page. This route was for testing. To test this module, explicitly run this flask app!!
def home():
    return render_template('churn.html')

@app.route('/predict', methods=['POST'])
def predict():
    # file = request.files['file']
    # file_path = os.path.join(os.getcwd(), file.filename)
    # file.save(file_path)

    # rfm, rfm_scaled = preprocess_data(file_path)
    # churn_preds = model.predict(rfm_scaled)

    # Integrating ANN model with Segmentation model
    rfm_scaled_path = os.path.join(os.path.dirname(__file__), '..', 'src/static', 'rfm_scaled_result.csv')
    rfm_scaled=pd.read_csv(rfm_scaled_path)
    rfm_path = os.path.join(os.path.dirname(__file__), '..', 'src/static', 'rfm_result.csv')
    rfm=pd.read_csv(rfm_path)
    churn_preds = model.predict(rfm_scaled)
    churn_labels = (churn_preds > 0.5).astype(int).flatten()
    rfm['Churn_Label'] = churn_labels

    rfm['Churn_Probability'] = churn_preds.flatten()  # flatten to 1D array
    # print(rfm.shape)

    # Saving CSV as backup (Optional)
    output_path = os.path.join(os.path.dirname(__file__), 'static', 'churn_results.csv')
    rfm.to_csv(output_path, index=False)

    # Converting dataframe to JSON and send
    result_json = rfm.to_dict(orient='records')
    return jsonify(result_json)

    # This part should be ignored
    # rfm['Churn_Label'] = churn_labels

    # rfm['Churn_Probability'] = churn_preds.flatten()  # flatten to 1D array
    # print(rfm.shape)

    # # Optionally save CSV as backup
    # output_path = os.path.join(os.path.dirname(__file__), 'static', 'churn_results.csv')
    # rfm.to_csv(output_path, index=False)

    # # Convert dataframe to JSON and send
    # result_json = rfm.to_dict(orient='records')
    # return jsonify(result_json)
