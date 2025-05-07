from dotenv import load_dotenv
load_dotenv()
from flask_cors import CORS
from flask import Flask, request, jsonify, render_template
import joblib  # Changed from pickle to joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json

app = Flask(__name__)
CORS(app)

model = joblib.load(os.path.join(os.path.dirname(__file__), 'kmeans_model.joblib'))
  # Modified to use joblib.load

# Define the static folder path relative to the current file
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


# Make sure the 'static' folder exists
os.makedirs(static_folder, exist_ok=True)


def load_and_clean_data(file_path):
  
    # Loads, cleans, and prepares data for RFM analysis and outlier removal. Args:file_path (str): The path to the CSV file. Returns: pandas.DataFrame: The cleaned and processed DataFrame.

    # Load data
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)

    # Convert CustomerID to string and create Amount column
    #retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

    # Compute RFM metrics
    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']

    # Corrected InvoiceDate format
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], dayfirst=True, errors='coerce')


    max_date = retail['InvoiceDate'].max()

    retail['Diff'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days

    rfm = pd.merge(rfm_m, rfm_f, on="CustomerID", how="inner")
    rfm = pd.merge(rfm, rfm_p, on="CustomerID", how="inner")
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

    # Remove outliers
    Q1 = rfm.quantile(0.05)
    Q3 = rfm.quantile(0.95)
    IQR = Q3 - Q1

    rfm = rfm[(rfm.Amount >= Q1[0] - 1.5 * IQR[0]) & (rfm.Amount <= Q3[0] + 1.5 * IQR[0])]
    rfm = rfm[(rfm.Recency >= Q1[2] - 1.5 * IQR[2]) & (rfm.Recency <= Q3[2] + 1.5 * IQR[2])]
    rfm = rfm[(rfm.Frequency >= Q1[1] - 1.5 * IQR[1]) & (rfm.Frequency <= Q3[1] + 1.5 * IQR[1])]

    return rfm
   


def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)

    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

    # Instantiate
    scaler = StandardScaler()

    # fit_transform
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)

    # rfm_df_scaled
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

    return rfm, rfm_df_scaled

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)

    file.save(file_path)
    df = preprocess_data(file_path)[1]
    results_df = model.predict(df)
    # results_df = pd.DataFrame(results_df)

    df_with_id = preprocess_data(file_path)[0]

    df_with_id['Cluster_Id'] = results_df


    # Generate the images and save them
    sns.stripplot(x='Cluster_Id', y='Amount', data=df_with_id, hue='Cluster_Id')
    amount_img_path = os.path.join(static_folder, 'ClusterId_Amount.png')
    plt.savefig(amount_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_Id', y='Frequency', data=df_with_id, hue='Cluster_Id')
    freq_img_path = os.path.join(static_folder, 'ClusterId_Frequency.png')
    plt.savefig(freq_img_path)
    plt.clf()

    sns.stripplot(x='Cluster_Id', y='Recency', data=df_with_id, hue='Cluster_Id')
    recency_img_path = os.path.join(static_folder, 'ClusterId_Recency.png')
    plt.savefig(recency_img_path)
    plt.clf()

    # Return the filenames of the generated images as a JSON response
    response = {
        'amount_img': 'static/ClusterId_Amount.png',  # This is the relative path you can use in your response
        'freq_img': 'static/ClusterId_Frequency.png',
        'recency_img': 'static/ClusterId_Recency.png'
    }

    return json.dumps(response)





