import streamlit as st
import joblib
import os
import numpy as np

# Load model
MODEL_PATH = "Model/pump_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def calculate_efficiency(flow, pressure):
    return (flow * pressure) / (flow + pressure)

st.set_page_config(page_title="Pump Failure Predictor", layout="centered")

st.title("🛠️ Pump Failure Predictor")
st.markdown("Enter the pump sensor readings to predict the probability of failure.")

# Input form
with st.form("prediction_form"):
    flow = st.number_input("Flow Rate", min_value=0.0, format="%.2f")
    pressure = st.number_input("Pressure", min_value=0.0, format="%.2f")
    vibration = st.number_input("Vibration", min_value=0.0, format="%.2f")
    temperature = st.number_input("Temperature", min_value=0.0, format="%.2f")
    
    submitted = st.form_submit_button("Predict")

if submitted:
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please train the model first.")
    else:
        model = load_model()
        X = np.array([[flow, pressure, vibration, temperature]])
        proba = model.predict_proba(X)[0][1]
        efficiency = calculate_efficiency(flow, pressure)

        st.success(f"🔍 Failure Probability: **{proba:.2%}**")
        st.info(f"⚙️ Efficiency: **{efficiency:.2f}**")
