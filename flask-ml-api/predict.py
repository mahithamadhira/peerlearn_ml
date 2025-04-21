# predict.py
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("xgb_reviewer_model.pkl")

# Feature order
FEATURES = [
    'reviewer_avg_error_all',
    'reviewer_review_count',
    'reviewer_std_dev_error',
    'reviewer_avg_error_on_assign',
    'reviewer_review_count_on_assign'
]

def predict_abs_error(features: dict) -> float:
    df = pd.DataFrame([features])[FEATURES]
    pred_log = model.predict(df)
    return float(np.expm1(pred_log))  # Inverse of log1p
