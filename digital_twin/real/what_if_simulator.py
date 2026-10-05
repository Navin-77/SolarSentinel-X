"""
SolarSentinel-X
What-If Digital Twin Simulation

Uses the REAL trained Random Forest and Autoencoder
to evaluate hypothetical sensor conditions.

IMPORTANT:
This module is simulation only.
It does not modify real ESP32 data or retrain models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

import math


class WhatIfSimulator:

    def __init__(self):

        project_root = Path(__file__).resolve().parents[2]

        # =====================================================
        # PATHS
        # =====================================================

        self.rf_model_path = (
            project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "random_forest_real.pkl"
        )

        self.rf_scaler_path = (
            project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "real_scaler.pkl"
        )

        self.health_encoder_path = (
            project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "health_encoder_real.pkl"
        )

        self.feature_names_path = (
            project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "feature_names_real.csv"
        )

        self.ae_model_path = (
            project_root
            / "models"
            / "autoencoder"
            / "real"
            / "autoencoder_real.keras"
        )

        self.ae_scaler_path = (
            project_root
            / "models"
            / "autoencoder"
            / "real"
            / "real_autoencoder_scaler.pkl"
        )

        self.ae_threshold_path = (
            project_root
            / "models"
            / "autoencoder"
            / "real"
            / "real_autoencoder_threshold.pkl"
        )

        # =====================================================
        # LOAD RANDOM FOREST
        # =====================================================

        self.rf_model = joblib.load(
            self.rf_model_path
        )

        self.rf_scaler = joblib.load(
            self.rf_scaler_path
        )

        self.health_encoder = joblib.load(
            self.health_encoder_path
        )

        self.feature_names = (
            pd.read_csv(
                self.feature_names_path
            )
            .iloc[:, 0]
            .dropna()
            .tolist()
        )

        # =====================================================
        # LOAD AUTOENCODER
        # =====================================================

        self.autoencoder = tf.keras.models.load_model(
            self.ae_model_path
        )

        self.ae_scaler = joblib.load(
            self.ae_scaler_path
        )

        self.ae_threshold = float(
            joblib.load(
                self.ae_threshold_path
            )
        )

        print(
            "✓ What-If Simulator initialized successfully."
        )

    # =========================================================
    # PREPARE INPUT
    # =========================================================

    def prepare_input(self, sensor_data):

        required_features = [
            "LDR",
            "DHT22_Temperature",
            "DHT22_Humidity",
            "DS18B20_Temperature",
            "Solar_Voltage",
            "Solar_Current",
            "Solar_Power",
            "Battery_Voltage",
            "Battery_Current",
            "Battery_Power",
        ]

        missing = [
            feature for feature in required_features
            if feature not in sensor_data
        ]

        if missing:
            raise ValueError(
                "Missing What-If sensor values: "
                + ", ".join(missing)
            )

        for feature in required_features:
            try:
                value = float(sensor_data[feature])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid What-If value for {feature}: "
                    f"{sensor_data[feature]!r}"
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"Invalid What-If value for {feature}: "
                    f"{value}. Please enter a finite value."
                )

        values = {
            "LDR": sensor_data["LDR"],
            "DHT22_Temperature": sensor_data[
                "DHT22_Temperature"
            ],
            "DHT22_Humidity": sensor_data[
                "DHT22_Humidity"
            ],
            "DS18B20_Temperature": sensor_data[
                "DS18B20_Temperature"
            ],
            "Solar_Voltage": sensor_data[
                "Solar_Voltage"
            ],
            "Solar_Current": sensor_data[
                "Solar_Current"
            ],
            "Solar_Power": sensor_data[
                "Solar_Power"
            ],
            "Battery_Voltage": sensor_data[
                "Battery_Voltage"
            ],
            "Battery_Current": sensor_data[
                "Battery_Current"
            ],
            "Battery_Power": sensor_data[
                "Battery_Power"
            ],
        }

        df = pd.DataFrame(
            [[values[name] for name in self.feature_names]],
            columns=self.feature_names
        )

        return df

    # =========================================================
    # RANDOM FOREST
    # =========================================================

    def predict_random_forest(self, X):

        X_scaled = self.rf_scaler.transform(
            X.values
        )

        prediction = self.rf_model.predict(
            X_scaled
        )

        probabilities = (
            self.rf_model.predict_proba(
                X_scaled
            )[0]
        )

        health = self.health_encoder.inverse_transform(
            prediction
        )[0]

        confidence = float(
            np.max(probabilities) * 100
        )

        probability_dict = {}

        for label, probability in zip(
            self.health_encoder.classes_,
            probabilities
        ):
            probability_dict[str(label)] = (
                float(probability)
            )

        return {
            "health": str(health),
            "confidence": confidence,
            "probabilities": probability_dict,
        }

    # =========================================================
    # AUTOENCODER
    # =========================================================

    def detect_anomaly(self, X):

        X_scaled = self.ae_scaler.transform(
            X.values
        )

        reconstructed = self.autoencoder.predict(
            X_scaled,
            verbose=0
        )

        reconstruction_error = float(
            np.mean(
                np.square(
                    X_scaled - reconstructed
                ),
                axis=1
            )[0]
        )

        if not np.isfinite(reconstruction_error):
            raise ValueError(
                "Autoencoder produced an invalid reconstruction error. "
                "Check the simulated sensor values and model inputs."
            )

        if reconstruction_error > self.ae_threshold:

            status = "Anomaly"

        else:

            status = "Normal"

        return {
            "status": status,
            "reconstruction_error":
                reconstruction_error,
            "threshold":
                self.ae_threshold,
        }

    # =========================================================
    # DECISION
    # =========================================================

    def calculate_decision(
        self,
        rf_health,
        anomaly_status
    ):

        if rf_health == "Critical":

            overall = "CRITICAL"

            recommendation = (
                "Immediate inspection and "
                "maintenance are recommended."
            )

            alert = (
                "CRITICAL: Simulated system "
                "condition requires immediate attention."
            )

        elif (
            rf_health == "Warning"
            or anomaly_status == "Anomaly"
        ):

            overall = "WARNING"

            recommendation = (
                "Inspect the simulated system "
                "and monitor sensor conditions."
            )

            alert = (
                "WARNING: Simulated abnormal "
                "condition detected."
            )

        else:

            overall = "NORMAL"

            recommendation = (
                "Simulated system is operating "
                "within the learned normal pattern."
            )

            alert = (
                "NORMAL: No simulated critical "
                "condition detected."
            )

        return {
            "overall_status": overall,
            "recommendation": recommendation,
            "alert": alert,
        }

    # =========================================================
    # COMPLETE SIMULATION
    # =========================================================

    def simulate(self, sensor_data):

        X = self.prepare_input(
            sensor_data
        )

        rf_result = self.predict_random_forest(
            X
        )

        ae_result = self.detect_anomaly(
            X
        )

        decision = self.calculate_decision(
            rf_result["health"],
            ae_result["status"]
        )

        return {

            "mode": "SIMULATION",

            "sensor_data": sensor_data,

            "random_forest": rf_result,

            "autoencoder": ae_result,

            "decision": decision,

            "lstm": {
                "status":
                    "Not evaluated",
                "reason":
                    "LSTM requires a temporal "
                    "10-sample sequence."
            },
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    simulator = WhatIfSimulator()

    test_condition = {

        "LDR": 500.0,

        "DHT22_Temperature": 32.0,

        "DHT22_Humidity": 70.0,

        "DS18B20_Temperature": 31.0,

        "Solar_Voltage": 1.0,

        "Solar_Current": 2.0,

        "Solar_Power": 5.0,

        "Battery_Voltage": 6.3,

        "Battery_Current": -0.5,

        "Battery_Power": 4.0,
    }

    result = simulator.simulate(
        test_condition
    )

    print("\n" + "=" * 70)
    print("SOLARSENTINEL-X WHAT-IF SIMULATION")
    print("=" * 70)

    print(
        "\nRF Health:",
        result["random_forest"]["health"]
    )

    print(
        "RF Confidence:",
        f"{result['random_forest']['confidence']:.2f}%"
    )

    print(
        "\nAutoencoder:",
        result["autoencoder"]["status"]
    )

    print(
        "Reconstruction Error:",
        f"{result['autoencoder']['reconstruction_error']:.6f}"
    )

    print(
        "Threshold:",
        f"{result['autoencoder']['threshold']:.6f}"
    )

    print(
        "\nOverall:",
        result["decision"]["overall_status"]
    )

    print(
        "Recommendation:",
        result["decision"]["recommendation"]
    )

    print("=" * 70)