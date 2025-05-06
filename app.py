import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class PumpMonitor:
    def __init__(self, data_path="data/pump_data.csv"):
        self.data = self._load_data(data_path)
        self.model = None
        self.accuracy = None

    def _load_data(self, path):
        try:
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except FileNotFoundError:
            st.error(f"File not found at: {path}")
            return None

    def calculate_efficiency(self, flow, pressure):
        return (flow * pressure) / (flow + pressure) if (flow + pressure) != 0 else 0

    def train_model(self, save_path="Model/pump_model.pkl"):
        if self.data is None:
            st.error("No data to train on.")
            return None

        X = self.data[['flow', 'pressure', 'vibration', 'temperature']]
        y = self.data['failure']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, save_path)
        return self.accuracy

    def predict_failure(self, flow, pressure, vibration, temperature):
        if self.model is None:
            try:
                self.model = joblib.load("Model/pump_model.pkl")
            except FileNotFoundError:
                st.error("Model not found. Train it first.")
                return None

        proba = self.model.predict_proba([[flow, pressure, vibration, temperature]])
        return proba[0][1]

# Streamlit UI
st.title("🚨 Pump Failure Prediction Dashboard")

monitor = PumpMonitor("data/pump_data.csv")

if monitor.data is not None:
    st.subheader("📊 Pump Data Sample")
    st.dataframe(monitor.data.head())

    if st.button("🛠 Train Model"):
        with st.spinner("Training..."):
            acc = monitor.train_model()
            if acc is not None:
                st.success(f"Model trained! Accuracy: {acc:.4f}")

    st.subheader("🔍 Predict Pump Failure")
    flow = st.number_input("Flow", min_value=0.0, value=100.0)
    pressure = st.number_input("Pressure", min_value=0.0, value=2.0)
    vibration = st.number_input("Vibration", min_value=0.0, value=0.5)
    temperature = st.number_input("Temperature", min_value=0.0, value=70.0)

    if st.button("Predict"):
        prob = monitor.predict_failure(flow, pressure, vibration, temperature)
        if prob is not None:
            efficiency = monitor.calculate_efficiency(flow, pressure)
            st.info(f"Failure Probability: {prob:.4f}")
            st.info(f"Efficiency: {efficiency:.2f}")
else:
    st.warning("⚠ Could not load data. Please check your CSV file path.")







