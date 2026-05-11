import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

st.set_page_config(page_title="Industrial Predictive Maintenance",page_icon="⚙️",layout="centered")

@st.cache_resource
def load_models():
    model=joblib.load('xgb_maintenance_model.joblib')
    scaler=joblib.load('feature_scaler.joblib')
    encoder=joblib.load('type_encoder.joblib')
    return model,scaler,encoder
model,scaler,encoder=load_models()
st.title("Equipment Failure Predictor")
st.markdown("Enter live telemetry data below to predict the likelihood of an imminent machine failure.")

with st.form("telemetry_form"):
    