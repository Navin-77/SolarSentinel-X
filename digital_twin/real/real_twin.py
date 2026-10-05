"""
SolarSentinel-X Real Digital Twin

Integrates the real ESP32 sensor data with the trained
real-sensor AI models.

Models used:
    - Real Random Forest
    - Real Autoencoder
    - Real LSTM

This module does NOT modify the existing synthetic-model
Digital Twin.
"""

from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from digital_twin.real.real_twin_state import RealTwinState


class RealDigitalTwin:

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]

        # --------------------------------------------------
        # REAL MODEL PATHS
        # --------------------------------------------------

        self.rf_model_path = (
            self.project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "random_forest_real.pkl"
        )

        self.rf_scaler_path = (
            self.project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "real_scaler.pkl"
        )

        self.rf_encoder_path = (
            self.project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "health_encoder_real.pkl"
        )

        self.rf_features_path = (
            self.project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "feature_names_real.csv"
        )

        self.ae_model_path = (
            self.project_root
            / "models"
            / "autoencoder"
            / "real"
            / "autoencoder_real.keras"
        )

        self.ae_scaler_path = (
            self.project_root
            / "models"
            / "autoencoder"
            / "real"
            / "real_autoencoder_scaler.pkl"
        )

        self.ae_threshold_path = (
            self.project_root
            / "models"
            / "autoencoder"
            / "real"
            / "real_autoencoder_threshold.pkl"
        )

        self.lstm_model_path = (
            self.project_root
            / "models"
            / "lstm"
            / "real"
            / "lstm_real_model.keras"
        )

        self.lstm_scaler_path = (
            self.project_root
            / "dataset"
            / "real_lstm"
            / "real_lstm_scaler.pkl"
        )

        self.lstm_mapping_path = (
            self.project_root
            / "dataset"
            / "real_lstm"
            / "real_lstm_label_mapping.pkl"
        )

        # --------------------------------------------------
        # LOAD MODELS
        # --------------------------------------------------

        print("=" * 60)
        print("SolarSentinel-X Real Digital Twin")
        print("=" * 60)

        print("\nLoading Real Random Forest...")
        self.rf_model = joblib.load(self.rf_model_path)
        self.rf_scaler = joblib.load(self.rf_scaler_path)
        self.rf_encoder = joblib.load(self.rf_encoder_path)

        self.rf_feature_names = (
            pd.read_csv(self.rf_features_path)
            .iloc[:, 0]
            .dropna()
            .tolist()
        )

        print("Real Random Forest loaded.")

        print("\nLoading Real Autoencoder...")
        self.autoencoder = load_model(self.ae_model_path)
        self.ae_scaler = joblib.load(self.ae_scaler_path)
        self.ae_threshold = float(joblib.load(self.ae_threshold_path))

        print("Real Autoencoder loaded.")

        print("\nLoading Real LSTM...")
        self.lstm_model = load_model(self.lstm_model_path)
        self.lstm_scaler = joblib.load(self.lstm_scaler_path)
        self.lstm_label_mapping = joblib.load(self.lstm_mapping_path)

        print("Real LSTM loaded.")

        # --------------------------------------------------
        # LSTM SEQUENCE BUFFER
        # --------------------------------------------------

        self.lstm_sequence_length = 10
        self.lstm_buffer = []

        # --------------------------------------------------
        # DIGITAL TWIN STATE
        # --------------------------------------------------

        self.state = RealTwinState()

        print("\nAll Real Digital Twin models loaded successfully.")
        print("=" * 60)

    # ======================================================
    # FEATURE PREPARATION
    # ======================================================

    def prepare_features(self, sensor_data):
        """
        Convert the real ESP32 sensor dictionary into the
        exact 10-feature model input.
        """

        feature_map = {
            "LDR": sensor_data["LDR"],
            "DHT22_Temperature": sensor_data["DHT22_Temperature"],
            "DHT22_Humidity": sensor_data["DHT22_Humidity"],
            "DS18B20_Temperature": sensor_data["DS18B20_Temperature"],
            "Solar_Voltage": sensor_data["Solar_Voltage"],
            "Solar_Current": sensor_data["Solar_Current"],
            "Solar_Power": sensor_data["Solar_Power"],
            "Battery_Voltage": sensor_data["Battery_Voltage"],
            "Battery_Current": sensor_data["Battery_Current"],
            "Battery_Power": sensor_data["Battery_Power"],
        }

        df = pd.DataFrame(
            [[feature_map[name] for name in self.rf_feature_names]],
            columns=self.rf_feature_names
        )

        return df

    # ======================================================
    # RANDOM FOREST
    # ======================================================

    def predict_health(self, features):
        """
        Predict system health using the real Random Forest.
        """

        scaled = self.rf_scaler.transform(features)

        prediction_encoded = self.rf_model.predict(scaled)[0]

        probabilities = self.rf_model.predict_proba(scaled)[0]

        prediction = self.rf_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        confidence = float(np.max(probabilities))

        probability_dict = {}

        for index, class_value in enumerate(
            self.rf_model.classes_
        ):
            class_name = self.rf_encoder.inverse_transform(
                [class_value]
            )[0]

            probability_dict[class_name] = float(
                probabilities[index]
            )

        return (
            prediction,
            confidence,
            probability_dict
        )

    # ======================================================
    # AUTOENCODER
    # ======================================================

    def detect_anomaly(self, features):
        """
        Detect unknown/anomalous sensor patterns using
        the real Autoencoder.
        """

        scaled = self.ae_scaler.transform(features)

        reconstructed = self.autoencoder.predict(
            scaled,
            verbose=0
        )

        reconstruction_error = float(
            np.mean(
                np.square(
                    scaled - reconstructed
                )
            )
        )

        if reconstruction_error > self.ae_threshold:
            status = "Anomaly"
        else:
            status = "Normal"

        return (
            status,
            reconstruction_error,
            self.ae_threshold
        )

    # ======================================================
    # LSTM
    # ======================================================

    def predict_temporal_health(self, features):
        """
        Update the LSTM sequence buffer and perform temporal
        health prediction when enough samples are available.

        The LSTM requires 10 consecutive real observations.
        """

        feature_values = features.iloc[0].values.astype(float)

        self.lstm_buffer.append(feature_values)

        if len(self.lstm_buffer) > self.lstm_sequence_length:
            self.lstm_buffer.pop(0)

        if len(self.lstm_buffer) < self.lstm_sequence_length:
            return None, None, None

        sequence = np.array(
            self.lstm_buffer,
            dtype=float
        )

        scaled_sequence = self.lstm_scaler.transform(
            sequence
        )

        lstm_input = np.expand_dims(
            scaled_sequence,
            axis=0
        )

        probabilities = self.lstm_model.predict(
            lstm_input,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        # Label mapping generated during real LSTM
        # sequence creation.
        if isinstance(self.lstm_label_mapping, dict):
            inverse_mapping = {
                value: key
                for key, value
                in self.lstm_label_mapping.items()
            }

            prediction = inverse_mapping.get(
                predicted_index,
                str(predicted_index)
            )
        else:
            prediction = str(predicted_index)

        confidence = float(
            np.max(probabilities)
        )

        probability_dict = {
            str(index): float(probability)
            for index, probability
            in enumerate(probabilities)
        }

        return (
            prediction,
            confidence,
            probability_dict
        )

    # ======================================================
    # COMPLETE UPDATE
    # ======================================================

    def update(self, sensor_data):
        """
        Process one real ESP32 sensor observation through
        the complete Real Digital Twin.
        """

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )

        features = self.prepare_features(
            sensor_data
        )

        # -------------------------------
        # Random Forest
        # -------------------------------

        (
            rf_prediction,
            rf_confidence,
            rf_probabilities
        ) = self.predict_health(features)

        # -------------------------------
        # Autoencoder
        # -------------------------------

        (
            anomaly_status,
            reconstruction_error,
            threshold
        ) = self.detect_anomaly(features)

        # -------------------------------
        # LSTM
        # -------------------------------

        (
            lstm_prediction,
            lstm_confidence,
            lstm_probabilities
        ) = self.predict_temporal_health(
            features
        )

        # -------------------------------
        # Update Digital Twin State
        # -------------------------------

        self.state.update_sensor_data(
            sensor_data
        )

        self.state.update_rf(
            rf_prediction,
            rf_confidence,
            rf_probabilities
        )

        self.state.update_autoencoder(
            anomaly_status,
            reconstruction_error,
            threshold
        )

        self.state.update_lstm(
            lstm_prediction,
            lstm_confidence,
            lstm_probabilities
        )

        self.state.update_timestamp(
            timestamp
        )

        return self.state.get_state()


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    test_sensor_data = {
        "LDR": 850.0,
        "DHT22_Temperature": 30.4,
        "DHT22_Humidity": 77.8,
        "DS18B20_Temperature": 29.87,
        "Solar_Voltage": 1.972,
        "Solar_Current": 7.1,
        "Solar_Power": 14.0,
        "Battery_Voltage": 0.0,
        "Battery_Current": -0.4,
        "Battery_Power": 0.0,
    }

    twin = RealDigitalTwin()

    print("\nRunning Real Digital Twin test...")

    for i in range(10):

        result = twin.update(
            test_sensor_data
        )

        print(
            f"\nSample {i + 1}"
        )

        print(
            "RF Health       :",
            result["random_forest"]["prediction"]
        )

        print(
            "RF Confidence   :",
            result["random_forest"]["confidence"]
        )

        print(
            "Autoencoder     :",
            result["autoencoder"]["status"]
        )

        print(
            "Reconstruction  :",
            result["autoencoder"]["reconstruction_error"]
        )

        print(
            "LSTM Health     :",
            result["lstm"]["prediction"]
        )

        print(
            "LSTM Confidence :",
            result["lstm"]["confidence"]
        )

    print("\nReal Digital Twin test completed.")