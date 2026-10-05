"""
Digital Twin

AI-powered virtual representation of the solar microgrid.
"""

import pandas as pd
from digital_twin.model_loader import ModelLoader

from digital_twin.utils import (
    load_latest_sensor_data,
    load_latest_dashboard_data,
    load_dashboard_history,
)
import numpy as np


class DigitalTwin:
    """
    Represents the complete solar microgrid digital twin.
    """

    def __init__(self):
        # Load AI models
        loader = ModelLoader()

        self.random_forest = loader.load_random_forest()
        self.shap_explainer = loader.load_shap_explainer(
            self.random_forest
        )
        self.lstm = loader.load_lstm()
        self.autoencoder = loader.load_autoencoder()

        self.threshold = loader.load_threshold()

        self.scaler = loader.load_scaler()
        self.health_encoder = loader.load_health_encoder()
        self.charging_encoder = loader.load_charging_encoder()

        # Load current sensor state
        self.current_state = load_latest_sensor_data()

        print("✓ Digital Twin initialized successfully.")
        
    def predict_current_health(self):
        """
        Predict the current health status of the solar microgrid.
        """

        # Predicted class
        prediction = self.random_forest.predict(self.current_state)

        # Class probabilities
        probabilities = self.random_forest.predict_proba(self.current_state)

        # Convert numeric label to text
        health_status = self.health_encoder.inverse_transform(prediction)[0]

        # Highest probability = confidence
        confidence = float(np.max(probabilities) * 100)

        print("\n========== Current System Health ==========")
        print(f"Predicted Health : {health_status}")
        print(f"Confidence       : {confidence:.2f}%")
        print("===========================================")

        return {
            "health": health_status,
            "confidence": confidence
        }
        
    def predict_future_battery(self):
        """
        Predict future Battery SOC, SOH and RUL using the trained LSTM model.
        """

        # Convert DataFrame to NumPy array
        sequence = self.current_state.to_numpy()

        # Add batch dimension
        sequence = np.expand_dims(sequence, axis=0)

        prediction = self.lstm.predict(sequence, verbose=0)

        # --------------------------------------------------
        # LSTM outputs are StandardScaler normalized values
        # Convert them back to engineering values
        # --------------------------------------------------

        predicted_soc_scaled = prediction[0][0]
        predicted_soh_scaled = prediction[0][1]
        predicted_rul_scaled = prediction[0][2]

        predicted_soc = (
            predicted_soc_scaled
            * self.scaler.scale_[14]
            + self.scaler.mean_[14]
        )

        predicted_soh = (
            predicted_soh_scaled
            * self.scaler.scale_[15]
            + self.scaler.mean_[15]
        )

        predicted_rul = (
            predicted_rul_scaled
            * self.scaler.scale_[16]
            + self.scaler.mean_[16]
        )

        print("\n========== Future Battery Prediction ==========")
        print(f"Predicted Battery SOC : {predicted_soc:.2f}%")
        print(f"Predicted Battery SOH : {predicted_soh:.2f}%")
        print(f"Predicted Battery RUL : {predicted_rul:.2f} cycles")
        print("===============================================")

        return {
            "Battery_SOC": float(predicted_soc),
            "Battery_SOH": float(predicted_soh),
            "Battery_RUL": float(predicted_rul),
        }
        
    def detect_unknown_fault(self):
        """
        Detect unknown faults using the trained Autoencoder.
        """

        import numpy as np

        # Autoencoder expects individual samples (not sequences)
        latest_sample = self.current_state.tail(1).to_numpy()

        reconstruction = self.autoencoder.predict(
            latest_sample,
            verbose=0
        )

        reconstruction_error = np.mean(
            np.square(latest_sample - reconstruction)
        )

        is_anomaly = reconstruction_error > self.threshold

        print("\n========== Unknown Fault Detection ==========")
        print(f"Reconstruction Error : {reconstruction_error:.6f}")
        print(f"Threshold            : {self.threshold:.6f}")
        print(f"Unknown Fault        : {is_anomaly}")
        print("=============================================")

        return {
            "reconstruction_error": float(reconstruction_error),
            "threshold": float(self.threshold),
            "unknown_fault": bool(is_anomaly),
        }
        
    def explain_prediction(self):
        """
        Explain the current Random Forest prediction using SHAP.
        """

        latest_sample = self.current_state.tail(1)

        shap_values = self.shap_explainer(latest_sample)

        prediction = self.random_forest.predict(latest_sample)[0]

        class_name = self.health_encoder.inverse_transform([prediction])[0]

        feature_importance = (
            abs(shap_values.values[0, :, prediction])
        )

        feature_names = latest_sample.columns.tolist()

        explanation = sorted(
            zip(feature_names, feature_importance),
            key=lambda x: x[1],
            reverse=True
        )

        print("\n========== SHAP Explanation ==========")
        print(f"Predicted Class : {class_name}")
        print("\nTop 5 Important Features:\n")

        for feature, value in explanation[:5]:
            print(f"{feature:<30} {value:.6f}")

        print("======================================")

        return explanation
    
    def get_latest_sensor_values(self):
        """
        Return the latest sensor readings for the dashboard.
        """

        latest = load_latest_dashboard_data()

        return {
            "Solar_Irradiance": float(latest["Solar_Irradiance"]),
            "Ambient_Temperature": float(latest["Ambient_Temperature"]),
            "Humidity": float(latest["Humidity"]),

            "Panel_Voltage": float(latest["Panel_Voltage"]),
            "Panel_Current": float(latest["Panel_Current"]),
            "Panel_Power": float(latest["Panel_Power"]),
            "Panel_Temperature": float(latest["Panel_Temperature"]),

            "Battery_Voltage": float(latest["Battery_Voltage"]),
            "Battery_Current": float(latest["Battery_Current"]),
            "Battery_Temperature": float(latest["Battery_Temperature"]),

            "Load_Power": float(latest["Load_Power"]),
            "Load_Current": float(latest["Load_Current"])
        }
        
    def get_dashboard_history(self):
        """
        Return the last 24 hours of engineering data.
        """

        return load_dashboard_history()
    
    def get_shap_importance(self):
        """
        Return the top SHAP feature importance for dashboard visualization.
        """

        shap_values = self.shap_explainer(self.current_state)

        importance = (
            abs(shap_values.values)
            .mean(axis=0)
            .mean(axis=1)
        )

        importance_df = (
            pd.DataFrame({
                "Feature": self.current_state.columns,
                "Importance": importance
            })
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(8)
        )

        return importance_df