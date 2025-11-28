from dotenv import load_dotenv
load_dotenv()
import sys, os
sys.path.append(os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')


from flask import Flask, request, render_template
import joblib  
import seaborn as sns
import matplotlib.pyplot as plt
import json
from .utils import preprocess_data
from churn_prediction.app_churn import churn_bp


app = Flask(__name__)
app.register_blueprint(churn_bp,url_prefix="/churn")

model = joblib.load(os.path.join(os.path.dirname(__file__), 'kmeans_model.joblib'))


static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')



os.makedirs(static_folder, exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)

    file.save(file_path)
    df_with_id, df = preprocess_data(file_path)
    results_df = model.predict(df)

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

    response = {
        'amount_img': 'static/ClusterId_Amount.png', 
        'freq_img': 'static/ClusterId_Frequency.png',
        'recency_img': 'static/ClusterId_Recency.png'
    }

    return json.dumps(response)




