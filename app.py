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
    st.subheader("Live Sensor Readings")

    col1,col2=st.columns(2)
    with col1:
        air_temp = st.number_input("Air temperature (K)", min_value=250.0, max_value=350.0, value=298.1)
        process_temp = st.number_input("Process temperature (K)", min_value=250.0, max_value=350.0, value=308.6)
        machine_type = st.selectbox("Machine Type", options=['L', 'M', 'H'])
        
    with col2:
        rotational_speed = st.number_input("Rotational speed (rpm)", min_value=1000, max_value=3000, value=1551)
        torque = st.number_input("Torque (Nm)", min_value=10.0, max_value=80.0, value=42.8)
        tool_wear = st.number_input("Tool wear (min)", min_value=0, max_value=300, value=0)
        
    submit_button = st.form_submit_button(label="Analyze Telemetry")

if submit_button:
    temp_diff=process_temp-air_temp
    power=rotational_speed*torque

    type_encoded=encoder.transform([machine_type])[0]
    input_data = pd.DataFrame([[
        type_encoded, 
        air_temp, 
        process_temp, 
        rotational_speed, 
        torque, 
        tool_wear, 
        temp_diff, 
        power
    ]], columns=[
        'Type', 
        'Air temperature K', 
        'Process temperature K', 
        'Rotational speed rpm', 
        'Torque Nm', 
        'Tool wear min', 
        'Temp_Diff', 
        'Power'
    ])

    input_scaled=scaler.transform(input_data)

    prediction=model.predict(input_scaled)[0]
    probability=model.predict_proba(input_scaled)[0][1]
    
    if prediction==1:
        st.error(f"WARNING HIGH RISK OF FAILURE DETECTED")
        st.write(f"The model predicts a breakdown with **{probability:.1%} certainty**.")
        st.write("Recommendation: Halt operations and schedule immediate diagnostic maintenance.")
    else:
        st.success(f"SYSTEM NORMAL")
        st.write(f"The machine is operating within safe parameters (Failure probability: {probability:.1%}).")
        st.write("Recommendation: Continue standard operations.")