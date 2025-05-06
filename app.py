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
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except FileNotFoundError:
            print(f"Error: File not found at: {path}")
            return None

    def calculate_efficiency(self, flow, pressure):
        return (flow * pressure) / (flow + pressure) if (flow + pressure) != 0 else 0

    def train_model(self, save_path="models/pump_model.pkl"):
        if self.data is None:
            print("Error: No data to train on.")
            return None

        # Check if required columns exist
        required_cols = ['flow', 'pressure', 'vibration', 'temperature', 'failure']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
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
                self.model = joblib.load("models/pump_model.pkl")
            except FileNotFoundError:
                print("Error: Model not found. Train it first.")
                return None

        proba = self.model.predict_proba([[flow, pressure, vibration, temperature]])
        return proba[0][1]

def main():
    print("🚨 Pump Failure Prediction System")
    print("---------------------------------")
    
    # Initialize the monitor
    monitor = PumpMonitor("data/pump_data.csv")
    
    if monitor.data is not None:
        print("\n📊 Pump Data Sample:")
        print(monitor.data.head())
        
        # For Pyodide/Jupyter environments, we'll use default values
        try:
            train = input("\n🛠 Would you like to train the model? (y/n): ").lower()
        except:
            print("\n🛠 Running in Pyodide environment - using default values")
            train = 'y'
            flow, pressure, vibration, temperature = 100.0, 2.0, 0.5, 70.0
        
        if train == 'y':
            print("Training...")
            acc = monitor.train_model()
            if acc is not None:
                print(f"Model trained! Accuracy: {acc:.4f}")
        
        if 'flow' not in locals():  # Only ask for inputs if not in Pyodide
            print("\n🔍 Predict Pump Failure")
            print("Enter the following parameters:")
            try:
                flow = float(input("Flow: "))
                pressure = float(input("Pressure: "))
                vibration = float(input("Vibration: "))
                temperature = float(input("Temperature: "))
            except:
                print("Using default values for prediction")
                flow, pressure, vibration, temperature = 100.0, 2.0, 0.5, 70.0
        
        prob = monitor.predict_failure(flow, pressure, vibration, temperature)
        if prob is not None:
            efficiency = monitor.calculate_efficiency(flow, pressure)
            print(f"\nResults:")
            print(f"Failure Probability: {prob:.4f}")
            print(f"Efficiency: {efficiency:.2f}")
    else:
        print("⚠ Could not load data. Please check your CSV file path.")

if __name__ == "__main__":
    main()
