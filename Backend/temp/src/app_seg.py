from dotenv import load_dotenv
load_dotenv()

from flask import Blueprint, render_template, request, jsonify, url_for
import joblib  # Changed from pickle to joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
from src.utils import preprocess_data

model = joblib.load(os.path.join(os.path.dirname(__file__), 'kmeans_model.joblib'))
  # Modified to use joblib.load

# Define static folder path (inside your src folder)
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# Make sure it exists
os.makedirs(static_folder, exist_ok=True)

seg_bp = Blueprint(
    "src",
    __name__,
    template_folder="templates",
    static_folder=static_folder
)


@seg_bp.route("/")
def segmentation_home():
    return render_template('index.html')

@seg_bp.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)

    file.save(file_path)
    df_with_id, df = preprocess_data(file_path)
    results_df = model.predict(df)
    # results_df = pd.DataFrame(results_df)

    # df_with_id = preprocess_data(file_path)[0]

    df_with_id['Cluster_Id'] = results_df
    output=os.path.join(os.path.dirname(__file__), 'static', 'segmentation_results.csv')
    df_with_id.to_csv(output, index=False)


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
    'amount_img': url_for("src.static", filename="ClusterId_Amount.png"),
    'freq_img': url_for("src.static", filename="ClusterId_Frequency.png"),
    'recency_img': url_for("src.static", filename="ClusterId_Recency.png")
}

    # print(response)

    return json.dumps(response)

