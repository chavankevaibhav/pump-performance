import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings
import plotly.express as px # Import Plotly Express

# --- PumpMonitor Class (Adapted for Model/ directory) ---
class PumpMonitor:
    def __init__(self):
        self.data = None
        self.model = None
        self.accuracy = None
        self.model_path = "Model/pump_model.pkl"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def load_data(self, uploaded_file_or_path):
        try:
            if isinstance(uploaded_file_or_path, str):
                df = pd.read_csv(uploaded_file_or_path)
            else:
                df = pd.read_csv(uploaded_file_or_path)

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.data = df
            return True
        except FileNotFoundError:
            st.error(f"Error: File not found at specified path.")
            self.data = None
            return False
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.data = None
            return False

    def calculate_efficiency(self, flow, pressure):
        return (flow * pressure) / (flow + pressure) if (flow + pressure) != 0 else 0

    def train_model(self):
        if self.data is None:
            st.error("Error: No data to train on. Please upload data first.")
            return None

        required_cols = ['flow', 'pressure', 'vibration', 'temperature', 'failure']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            st.error(f"Error: Missing required columns in the uploaded data: {missing_cols}")
            return None
        
        if self.data['failure'].nunique() < 2:
            st.error("Error: The 'failure' column must have at least two distinct classes for training.")
            return None

        X = self.data[required_cols[:-1]]
        y = self.data['failure']

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            st.warning("Could not stratify data (likely due to imbalance in small dataset). Using regular split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        joblib.dump(self.model, self.model_path)
        st.session_state.model_trained_this_session = True
        st.session_state.trained_model_accuracy = self.accuracy
        return self.accuracy

    def predict_failure(self, flow, pressure, vibration, temperature):
        if self.model is None:
            try:
                self.model = joblib.load(self.model_path)
                st.info(f"Loaded pre-trained model from {self.model_path}")
            except FileNotFoundError:
                st.error(f"Error: Model not found. Train it first or ensure '{self.model_path}' exists.")
                return None
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
        
        if self.model is None:
             st.error("Model is not available for prediction.")
             return None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
            try:
                proba = self.model.predict_proba([[flow, pressure, vibration, temperature]])
                return proba[0][1] 
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                return None

# --- Streamlit App Logic ---
def main():
    st.set_page_config(page_title="Pump Failure Predictor", page_icon="🚨", layout="wide")
    st.title("🚨 Pump Failure Prediction System")
    st.markdown("---")

    if 'monitor' not in st.session_state:
        st.session_state.monitor = PumpMonitor()
    monitor = st.session_state.monitor

    if 'model_trained_this_session' not in st.session_state:
        st.session_state.model_trained_this_session = False
    if 'trained_model_accuracy' not in st.session_state:
        st.session_state.trained_model_accuracy = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    st.sidebar.header("⚙️ Configuration & Input")

    st.sidebar.subheader("1. Load Data")
    uploaded_file = st.sidebar.file_uploader("Upload your pump data CSV", type="csv")
    
    default_data_path = "data/pump_data.csv"
    
    if uploaded_file:
        if monitor.load_data(uploaded_file):
            st.session_state.data_loaded = True
            st.session_state.model_trained_this_session = False 
            st.session_state.trained_model_accuracy = None
            st.sidebar.success("Data loaded successfully from uploaded file!")
    elif not st.session_state.data_loaded and os.path.exists(default_data_path):
        if st.sidebar.button("Load Default Data Sample"):
            if monitor.load_data(default_data_path):
                st.session_state.data_loaded = True
                st.session_state.model_trained_this_session = False
                st.session_state.trained_model_accuracy = None
                st.sidebar.success(f"Default data loaded from '{default_data_path}'!")
            else:
                st.sidebar.error("Failed to load default data.")
    
    if st.session_state.data_loaded and monitor.data is not None:
        st.subheader("📊 Pump Data Sample")
        st.dataframe(monitor.data.head())

        st.subheader("🔬 Data Visualizations")
        plot_cols = ['flow', 'temperature', 'pressure']
        
        if all(col in monitor.data.columns for col in plot_cols):
            st.markdown("#### 3D Scatter Plot: Flow vs Temperature vs Pressure")
            try:
                fig_3d = px.scatter_3d(
                    monitor.data,
                    x='flow',
                    y='temperature',
                    z='pressure',
                    color='failure' if 'failure' in monitor.data.columns else None,
                    title="Pump Parameters (3D View)",
                    labels={'flow': 'Flow', 'temperature': 'Temperature', 'pressure': 'Pressure'},
                    hover_data=monitor.data.columns
                )
                fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig_3d, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate 3D plot: {e}")

            st.markdown("#### 2D Scatter Plots")
            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                try:
                    fig_flow_temp = px.scatter(
                        monitor.data, x='flow', y='temperature',
                        color='failure' if 'failure' in monitor.data.columns else None,
                        # Trendline requires statsmodels: pip install statsmodels
                        trendline="ols" if 'failure' in monitor.data.columns and monitor.data['failure'].nunique() > 1 else None,
                        title="Flow vs Temperature",
                        labels={'flow': 'Flow', 'temperature': 'Temperature'}
                    )
                    st.plotly_chart(fig_flow_temp, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate Flow vs Temperature plot: {e}")
                
                try:
                    fig_temp_pressure = px.scatter(
                        monitor.data, x='temperature', y='pressure',
                        color='failure' if 'failure' in monitor.data.columns else None,
                        # Trendline requires statsmodels: pip install statsmodels
                        trendline="ols" if 'failure' in monitor.data.columns and monitor.data['failure'].nunique() > 1 else None,
                        title="Temperature vs Pressure",
                        labels={'temperature': 'Temperature', 'pressure': 'Pressure'}
                    )
                    st.plotly_chart(fig_temp_pressure, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate Temperature vs Pressure plot: {e}")

            with plot_col2:
                try:
                    fig_flow_pressure = px.scatter(
                        monitor.data, x='flow', y='pressure',
                        color='failure' if 'failure' in monitor.data.columns else None,
                        # Trendline requires statsmodels: pip install statsmodels
                        trendline="ols" if 'failure' in monitor.data.columns and monitor.data['failure'].nunique() > 1 else None,
                        title="Flow vs Pressure",
                        labels={'flow': 'Flow', 'pressure': 'Pressure'}
                    )
                    st.plotly_chart(fig_flow_pressure, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate Flow vs Pressure plot: {e}")
                
                if 'vibration' in monitor.data.columns:
                    try:
                        fig_flow_vibration = px.scatter(
                            monitor.data, x='flow', y='vibration',
                            color='failure' if 'failure' in monitor.data.columns else None,
                            # Trendline requires statsmodels: pip install statsmodels
                            trendline="ols" if 'failure' in monitor.data.columns and monitor.data['failure'].nunique() > 1 else None,
                            title="Flow vs Vibration",
                            labels={'flow': 'Flow', 'vibration': 'Vibration'}
                        )
                        st.plotly_chart(fig_flow_vibration, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate Flow vs Vibration plot: {e}")
        else:
            missing_for_plot = [col for col in plot_cols if col not in monitor.data.columns]
            st.warning(f"Visualizations require columns: {', '.join(plot_cols)}. Missing: {', '.join(missing_for_plot)}")

        st.sidebar.subheader("2. Train Model")
        if st.sidebar.button("🛠️ Train Model on Current Data"):
            with st.spinner("Training in progress..."):
                acc = monitor.train_model()
                if acc is not None:
                    st.sidebar.success(f"Model trained! Accuracy: {acc:.4f}")
                else:
                    st.sidebar.error("Model training failed. Check data.")
    else:
        st.info("Please upload a CSV file or load the default data to proceed.")
        st.sidebar.markdown("_(Training and Visualizations disabled until data is loaded)_")

    st.sidebar.markdown("---")

    st.sidebar.subheader("3. Predict Failure")
    if not st.session_state.data_loaded and not os.path.exists(monitor.model_path):
         st.sidebar.warning("Upload data and train a model, or ensure a pre-trained model exists to enable prediction.")

    can_predict = st.session_state.model_trained_this_session or os.path.exists(monitor.model_path)

    if can_predict:
        st.sidebar.markdown("Enter pump parameters:")
        flow = st.sidebar.number_input("Flow (e.g., 100.0)", value=100.0, format="%.1f")
        pressure = st.sidebar.number_input("Pressure (e.g., 2.0)", value=2.0, format="%.1f")
        vibration = st.sidebar.number_input("Vibration (e.g., 0.5)", value=0.5, format="%.2f")
        temperature = st.sidebar.number_input("Temperature (e.g., 70.0)", value=70.0, format="%.1f")

        if st.sidebar.button("🔍 Predict"):
            if monitor.model is None and os.path.exists(monitor.model_path): 
                monitor.model = joblib.load(monitor.model_path)

            if monitor.model is not None:
                prob = monitor.predict_failure(flow, pressure, vibration, temperature)
                if prob is not None:
                    efficiency = monitor.calculate_efficiency(flow, pressure)
                    
                    st.subheader("📈 Prediction Results")
                    col1, col2 = st.columns(2)
                    col1.metric("Failure Probability", f"{prob:.4f}")
                    col2.metric("Calculated Efficiency", f"{efficiency:.2f}")

                    if prob > 0.75:
                        st.error(f"🚨 HIGH RISK of Pump Failure (Probability: {prob:.2%})")
                    elif prob > 0.5:
                        st.warning(f"⚠️ Moderate Risk of Pump Failure (Probability: {prob:.2%})")
                    else:
                        st.success(f"✅ Low Risk of Pump Failure (Probability: {prob:.2%})")
                else:
                    st.error("Prediction could not be made. Check model and inputs.")
            else:
                st.error(f"Model not available. Please train a model or ensure '{monitor.model_path}' exists.")
    else:
        st.sidebar.info("Train a model or ensure a pre-trained model exists to enable prediction.")
        
    if st.session_state.trained_model_accuracy is not None:
        st.sidebar.metric(label="Current Model Accuracy", value=f"{st.session_state.trained_model_accuracy:.4f}")
    elif os.path.exists(monitor.model_path) and monitor.model is None: 
        st.sidebar.info("A pre-trained model might be available. Load data and predict to use it, or retrain.")

if __name__ == "__main__":
    main()
