import os
from collections import deque

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# ============================================================
# SOLARSENTINEL-X
# REAL MODEL INFERENCE LAYER
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


# ============================================================
# REAL SENSOR FEATURES
# ============================================================

FEATURE_NAMES = [
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


# ============================================================
# MODEL PATHS
# ============================================================

RF_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest",
    "real_trained",
    "random_forest_real.pkl"
)

RF_SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest",
    "real_trained",
    "real_scaler.pkl"
)

RF_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest",
    "real_trained",
    "health_encoder_real.pkl"
)


AE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder",
    "real",
    "autoencoder_real.keras"
)

AE_SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder",
    "real",
    "real_autoencoder_scaler.pkl"
)

AE_THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder",
    "real",
    "real_autoencoder_threshold.pkl"
)


LSTM_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lstm",
    "real",
    "lstm_real_model.keras"
)

LSTM_SCALER_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real_lstm",
    "real_lstm_scaler.pkl"
)

LSTM_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real_lstm",
    "real_lstm_label_mapping.pkl"
)


# ============================================================
# LSTM CONFIGURATION
# ============================================================

LSTM_SEQUENCE_LENGTH = 10


# ============================================================
# REAL MODEL INFERENCE CLASS
# ============================================================

class RealModelInference:

    def __init__(self):

        print("\n" + "=" * 70)
        print("SOLARSENTINEL-X REAL MODEL INFERENCE")
        print("=" * 70)

        # ----------------------------------------------------
        # Load Random Forest
        # ----------------------------------------------------

        print("\nLoading Real Random Forest...")

        self.rf_model = joblib.load(
            RF_MODEL_PATH
        )

        self.rf_scaler = joblib.load(
            RF_SCALER_PATH
        )

        self.rf_encoder = joblib.load(
            RF_ENCODER_PATH
        )

        print("Real Random Forest loaded successfully.")

        # ----------------------------------------------------
        # Load Autoencoder
        # ----------------------------------------------------

        print("\nLoading Real Autoencoder...")

        self.ae_model = load_model(
            AE_MODEL_PATH
        )

        self.ae_scaler = joblib.load(
            AE_SCALER_PATH
        )

        self.ae_threshold = float(
            joblib.load(AE_THRESHOLD_PATH)
        )

        print("Real Autoencoder loaded successfully.")

        print(
            "Autoencoder threshold:",
            round(self.ae_threshold, 6)
        )

        # ----------------------------------------------------
        # Load LSTM
        # ----------------------------------------------------

        print("\nLoading Real LSTM...")

        self.lstm_model = load_model(
            LSTM_MODEL_PATH
        )

        self.lstm_scaler = joblib.load(
            LSTM_SCALER_PATH
        )

        self.lstm_label_mapping = joblib.load(
            LSTM_MAPPING_PATH
        )

        print("Real LSTM loaded successfully.")

        # ----------------------------------------------------
        # Sequence Buffer
        # ----------------------------------------------------

        self.sequence_buffer = deque(
            maxlen=LSTM_SEQUENCE_LENGTH
        )

        print(
            "\nLSTM sequence length:",
            LSTM_SEQUENCE_LENGTH
        )

        print("\nAll real models loaded successfully.")


    # ========================================================
    # VALIDATE SENSOR DATA
    # ========================================================

    def validate_sensor_data(self, sensor_data):

        missing = [
            feature
            for feature in FEATURE_NAMES
            if feature not in sensor_data
        ]

        if missing:

            raise ValueError(
                "Missing sensor features: "
                + ", ".join(missing)
            )

        values = []

        for feature in FEATURE_NAMES:

            try:

                value = float(
                    sensor_data[feature]
                )

            except (TypeError, ValueError):

                raise ValueError(
                    f"Invalid value for {feature}: "
                    f"{sensor_data[feature]}"
                )

            if not np.isfinite(value):

                raise ValueError(
                    f"Invalid numeric value for {feature}: "
                    f"{value}"
                )

            values.append(value)

        return np.array(
            values,
            dtype=float
        )


    # ========================================================
    # RANDOM FOREST PREDICTION
    # ========================================================

    def predict_random_forest(self, values):

        X = pd.DataFrame(
            [values],
            columns=FEATURE_NAMES
        )

        X_scaled = self.rf_scaler.transform(X)

        prediction = self.rf_model.predict(
            X_scaled
        )[0]

        probabilities = self.rf_model.predict_proba(
            X_scaled
        )[0]

        label = self.rf_encoder.inverse_transform(
            [int(prediction)]
        )[0]

        probability_dict = {}

        for index, class_name in enumerate(
            self.rf_encoder.classes_
        ):

            probability_dict[str(class_name)] = float(
                probabilities[index]
            )

        return {
            "prediction": str(label),
            "encoded_prediction": int(prediction),
            "probabilities": probability_dict,
            "confidence": float(
                np.max(probabilities)
            ),
        }


    # ========================================================
    # AUTOENCODER ANOMALY DETECTION
    # ========================================================

    def predict_autoencoder(self, values):

        X = pd.DataFrame(
            [values],
            columns=FEATURE_NAMES
        )

        X_scaled = self.ae_scaler.transform(X)

        reconstruction = self.ae_model.predict(
            X_scaled,
            verbose=0
        )

        reconstruction_error = float(
            np.mean(
                np.square(
                    X_scaled - reconstruction
                ),
                axis=1
            )[0]
        )

        is_anomaly = (
            reconstruction_error
            > self.ae_threshold
        )

        return {
            "reconstruction_error":
                reconstruction_error,

            "threshold":
                self.ae_threshold,

            "status":
                "Anomaly"
                if is_anomaly
                else "Normal",

            "is_anomaly":
                bool(is_anomaly),
        }


    # ========================================================
    # LSTM PREDICTION
    # ========================================================

    def predict_lstm(self, values):

        # Add current observation
        self.sequence_buffer.append(
            values.copy()
        )

        # Need 10 observations
        if len(self.sequence_buffer) < LSTM_SEQUENCE_LENGTH:

            return {
                "status": "Waiting",
                "ready": False,
                "samples_collected":
                    len(self.sequence_buffer),
                "samples_required":
                    LSTM_SEQUENCE_LENGTH,
            }

        # Convert sequence to array
        sequence = np.array(
            self.sequence_buffer,
            dtype=float
        )

        # Scale using the scaler used during training
        sequence_scaled = self.lstm_scaler.transform(
            sequence
        )

        # Add batch dimension
        X_sequence = np.expand_dims(
            sequence_scaled,
            axis=0
        )

        prediction = self.lstm_model.predict(
            X_sequence,
            verbose=0
        )

        probabilities = prediction[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        # Handle dictionary or list mapping
        label = self.decode_lstm_label(
            predicted_index
        )

        return {
            "status": "Ready",
            "ready": True,
            "prediction": label,
            "encoded_prediction": predicted_index,
            "probabilities": probabilities.tolist(),
            "confidence": float(
                np.max(probabilities)
            ),
            "sequence_length":
                LSTM_SEQUENCE_LENGTH,
        }


    # ========================================================
    # LSTM LABEL DECODER
    # ========================================================

    def decode_lstm_label(self, index):

        mapping = self.lstm_label_mapping

        # Dictionary mapping
        if isinstance(mapping, dict):

            # Try integer key
            if index in mapping:
                return str(mapping[index])

            # Try string key
            if str(index) in mapping:
                return str(mapping[str(index)])

            # Reverse mapping case
            for key, value in mapping.items():

                if value == index:
                    return str(key)

                if str(value) == str(index):
                    return str(key)

        # List / tuple mapping
        if isinstance(mapping, (list, tuple)):

            if 0 <= index < len(mapping):
                return str(mapping[index])

        return f"Class_{index}"


    # ========================================================
    # COMPLETE REAL INFERENCE
    # ========================================================

    def predict(self, sensor_data):

        values = self.validate_sensor_data(
            sensor_data
        )

        rf_result = self.predict_random_forest(
            values
        )

        ae_result = self.predict_autoencoder(
            values
        )

        lstm_result = self.predict_lstm(
            values
        )

        return {
            "sensor_data": {
                feature: float(values[index])
                for index, feature
                in enumerate(FEATURE_NAMES)
            },

            "random_forest": rf_result,

            "autoencoder": ae_result,

            "lstm": lstm_result,
        }


# ============================================================
# TEST FUNCTION
# ============================================================

if __name__ == "__main__":

    print("\nCreating RealModelInference...")

    inference = RealModelInference()

    # --------------------------------------------------------
    # Example REAL-format sensor record
    #
    # This is only a structural test.
    # It is NOT used for training.
    # --------------------------------------------------------

    test_sensor_data = {

        "LDR": 740.0,

        "DHT22_Temperature": 32.4,

        "DHT22_Humidity": 64.4,

        "DS18B20_Temperature": 31.0,

        "Solar_Voltage": 0.0,

        "Solar_Current": -0.3,

        "Solar_Power": 0.0,

        "Battery_Voltage": 8.096,

        "Battery_Current": -0.4,

        "Battery_Power": 0.0,
    }


    print("\n" + "=" * 70)
    print("TESTING REAL MODEL INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Feed 10 samples because LSTM requires a sequence
    # --------------------------------------------------------

    for sample_number in range(1, 11):

        result = inference.predict(
            test_sensor_data
        )

        print(
            f"\nSample {sample_number}/10"
        )

        print(
            "Random Forest:",
            result["random_forest"]["prediction"]
        )

        print(
            "RF Confidence:",
            round(
                result["random_forest"]["confidence"],
                4
            )
        )

        print(
            "Autoencoder:",
            result["autoencoder"]["status"]
        )

        print(
            "Reconstruction Error:",
            round(
                result["autoencoder"][
                    "reconstruction_error"
                ],
                6
            )
        )

        if result["lstm"]["ready"]:

            print(
                "LSTM:",
                result["lstm"]["prediction"]
            )

            print(
                "LSTM Confidence:",
                round(
                    result["lstm"]["confidence"],
                    4
                )
            )

        else:

            print(
                "LSTM:",
                "Waiting for sequence "
                f"({result['lstm']['samples_collected']}/"
                f"{result['lstm']['samples_required']})"
            )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REAL MODEL INFERENCE TEST COMPLETED")
    print("=" * 70)

    print("\nFinal result:")
    print(result)