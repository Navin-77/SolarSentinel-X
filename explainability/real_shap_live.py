"""
SolarSentinel-X Live SHAP Explanation

Generates SHAP explanations when the SHAP runtime is available.

If SHAP/Numba/llvmlite cannot be loaded, the dashboard continues
using the real Random Forest prediction without live SHAP values.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class RealSHAPExplainer:

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[1]

        # --------------------------------------------------
        # REAL RANDOM FOREST ARTIFACTS
        # --------------------------------------------------

        self.model_path = (
            self.project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "random_forest_real.pkl"
        )

        self.scaler_path = (
            self.project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "real_scaler.pkl"
        )

        self.encoder_path = (
            self.project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "health_encoder_real.pkl"
        )

        self.features_path = (
            self.project_root
            / "models"
            / "random_forest"
            / "real_trained"
            / "feature_names_real.csv"
        )

        # --------------------------------------------------
        # LOAD REAL MODEL ARTIFACTS
        # --------------------------------------------------

        print("\nLoading Real SHAP Explainer...")

        self.model = joblib.load(self.model_path)

        self.scaler = joblib.load(self.scaler_path)

        self.encoder = joblib.load(self.encoder_path)

        self.feature_names = (
            pd.read_csv(self.features_path)
            .iloc[:, 0]
            .dropna()
            .tolist()
        )

        # --------------------------------------------------
        # OPTIONAL SHAP
        # --------------------------------------------------

        self.explainer = None
        self.shap_available = False

        try:

            import shap

            self.explainer = shap.TreeExplainer(
                self.model
            )

            self.shap_available = True

            print(
                "Real SHAP Explainer loaded successfully."
            )

        except Exception as e:

            print(
                "\nWARNING: Live SHAP is temporarily unavailable."
            )

            print(
                "Random Forest inference will continue normally."
            )

            print(
                f"SHAP runtime error: {e}"
            )

    # ======================================================
    # PREPARE INPUT
    # ======================================================

    def prepare_input(self, sensor_data):

        values = [
            sensor_data[feature]
            for feature in self.feature_names
        ]

        return pd.DataFrame(
            [values],
            columns=self.feature_names
        )

    # ======================================================
    # EXPLAIN CURRENT PREDICTION
    # ======================================================

    def explain(self, sensor_data):

        features = self.prepare_input(sensor_data)

        # --------------------------------------------------
        # REAL RANDOM FOREST PREDICTION
        # --------------------------------------------------

        scaled_features = self.scaler.transform(
            features
        )

        prediction_encoded = self.model.predict(
            scaled_features
        )[0]

        prediction = self.encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        # --------------------------------------------------
        # FALLBACK IF SHAP IS UNAVAILABLE
        # --------------------------------------------------

        if not self.shap_available:

            return {
                "prediction": prediction,
                "prediction_encoded": int(
                    prediction_encoded
                ),
                "top_features": [],
                "all_features": [],
                "shap_available": False
            }

        # --------------------------------------------------
        # REAL SHAP VALUES
        # --------------------------------------------------

        shap_values = self.explainer.shap_values(
            scaled_features
        )

        shap_values = np.asarray(
            shap_values
        )

        # --------------------------------------------------
        # NORMALIZE SHAP OUTPUT
        # --------------------------------------------------

        if shap_values.ndim == 3:

            if shap_values.shape[0] == len(
                self.model.classes_
            ):

                class_shap = shap_values[
                    :, 0, :
                ].T

            elif shap_values.shape[2] == len(
                self.model.classes_
            ):

                class_shap = shap_values[
                    0, :, :
                ]

            else:

                raise ValueError(
                    "Unexpected SHAP array shape: "
                    f"{shap_values.shape}"
                )

        elif shap_values.ndim == 2:

            class_shap = shap_values

        else:

            raise ValueError(
                "Unexpected SHAP dimensions: "
                f"{shap_values.shape}"
            )

        # --------------------------------------------------
        # SELECT PREDICTED CLASS
        # --------------------------------------------------

        class_index = list(
            self.model.classes_
        ).index(
            prediction_encoded
        )

        if shap_values.ndim == 3:

            feature_shap_values = class_shap[
                :, class_index
            ]

        else:

            feature_shap_values = class_shap[:, 0]

        # --------------------------------------------------
        # BUILD EXPLANATION
        # --------------------------------------------------

        explanation = pd.DataFrame({
            "Feature": self.feature_names,
            "SHAP_Value": feature_shap_values
        })

        explanation["Absolute_SHAP"] = (
            explanation["SHAP_Value"].abs()
        )

        explanation = explanation.sort_values(
            "Absolute_SHAP",
            ascending=False
        ).reset_index(drop=True)

        # --------------------------------------------------
        # TOP FEATURES
        # --------------------------------------------------

        top_features = []

        for _, row in explanation.head(5).iterrows():

            influence = (
                "Positive"
                if row["SHAP_Value"] > 0
                else "Negative"
            )

            top_features.append({
                "feature": row["Feature"],
                "shap_value": float(
                    row["SHAP_Value"]
                ),
                "absolute_shap": float(
                    row["Absolute_SHAP"]
                ),
                "influence": influence
            })

        return {
            "prediction": prediction,
            "prediction_encoded": int(
                prediction_encoded
            ),
            "top_features": top_features,
            "all_features": explanation.to_dict(
                orient="records"
            ),
            "shap_available": True
        }


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

    explainer = RealSHAPExplainer()

    result = explainer.explain(
        test_sensor_data
    )

    print("\n" + "=" * 60)
    print("LIVE SHAP EXPLANATION")
    print("=" * 60)

    print(
        f"\nPredicted Health : "
        f"{result['prediction']}"
    )

    if result["shap_available"]:

        print("\nTop Influential Features:")

        for item in result["top_features"]:

            print(
                f"{item['feature']:25s} "
                f"SHAP = {item['shap_value']:+.6f} "
                f"({item['influence']})"
            )

    else:

        print(
            "\nSHAP calculation unavailable."
        )

        print(
            "Random Forest prediction is still active."
        )

    print("\nLive SHAP test completed.")