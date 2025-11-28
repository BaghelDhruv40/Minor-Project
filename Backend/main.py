from flask import Flask
from src.app_seg import seg_bp
# from churn.app_churn import churn_bp

app = Flask(__name__)


app.register_blueprint(seg_bp, url_prefix="/src")
# app.register_blueprint(churn_bp, url_prefix="/churn_prediction")

