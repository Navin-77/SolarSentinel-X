"""
SolarSentinel-X
Predictive Maintenance - Sensor Pattern Analyzer

Purpose:
    Compare incoming real-time sensor values with the patterns
    learned from the real training dataset.

This module does NOT claim a physical component has failed.
It identifies uncommon sensor behaviour and provides evidence
for the maintenance layer.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "real_training"
    / "real_sensor_dataset.csv"
)


# ============================================================
# SENSOR FEATURES
# ============================================================

FEATURES = [
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
# ANALYZER
# ============================================================

class SensorPatternAnalyzer:

    def __init__(self, dataset_path=DATASET_PATH):

        self.dataset_path = Path(dataset_path)

        self.dataset = None
        self.statistics = {}

        self._load_dataset()
        self._calculate_statistics()


    # ========================================================
    # LOAD DATASET
    # ========================================================

    def _load_dataset(self):

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                f"Real sensor dataset not found:\n"
                f"{self.dataset_path}"
            )

        self.dataset = pd.read_csv(self.dataset_path)

        missing = [
            feature
            for feature in FEATURES
            if feature not in self.dataset.columns
        ]

        if missing:

            raise ValueError(
                f"Missing sensor features: {missing}"
            )

        self.dataset[FEATURES] = self.dataset[
            FEATURES
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        self.dataset = self.dataset.dropna(
            subset=FEATURES
        ).reset_index(drop=True)


    # ========================================================
    # CALCULATE TRAINING PATTERNS
    # ========================================================

    def _calculate_statistics(self):

        for feature in FEATURES:

            values = self.dataset[feature].values.astype(float)

            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            mean = np.mean(values)
            std = np.std(values)

            if std == 0:
                std = 1e-9

            self.statistics[feature] = {
                "mean": mean,
                "std": std,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": lower,
                "upper": upper,
                "min": np.min(values),
                "max": np.max(values),
            }


    # ========================================================
    # ANALYZE ONE SENSOR VALUE
    # ========================================================

    def analyze_sensor(self, feature, value):

        if feature not in self.statistics:

            raise ValueError(
                f"Unknown sensor feature: {feature}"
            )

        stats = self.statistics[feature]

        try:
            value = float(value)

        except (TypeError, ValueError):

            return {
                "sensor": feature,
                "value": value,
                "status": "MISSING",
                "severity": "CRITICAL",
                "z_score": None,
                "deviation_percent": None,
                "message": "Sensor value is missing or invalid.",
            }


        mean = stats["mean"]
        std = stats["std"]

        z_score = abs(value - mean) / std

        # ----------------------------------------------------
        # Percentage deviation from learned mean
        # ----------------------------------------------------

        if abs(mean) > 1e-9:

            deviation_percent = (
                abs(value - mean)
                / abs(mean)
            ) * 100

        else:

            deviation_percent = 0.0


        # ----------------------------------------------------
        # Pattern classification
        # ----------------------------------------------------

        if value < stats["lower"] or value > stats["upper"]:

            status = "UNUSUAL"

        elif z_score >= 2:

            status = "DEVIATING"

        else:

            status = "NORMAL"


        # ----------------------------------------------------
        # Severity
        # ----------------------------------------------------

        if status == "UNUSUAL":

            severity = "HIGH"

        elif status == "DEVIATING":

            severity = "MEDIUM"

        else:

            severity = "LOW"


        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        if status == "UNUSUAL":

            message = (
                f"{feature} is outside the learned "
                f"operating pattern."
            )

        elif status == "DEVIATING":

            message = (
                f"{feature} is deviating from its "
                f"learned operating pattern."
            )

        else:

            message = (
                f"{feature} is within the learned "
                f"operating pattern."
            )


        return {
            "sensor": feature,
            "value": value,
            "status": status,
            "severity": severity,
            "z_score": round(float(z_score), 3),
            "deviation_percent": round(
                float(deviation_percent),
                2
            ),
            "message": message,
        }


    # ========================================================
    # ANALYZE COMPLETE SENSOR READING
    # ========================================================

    def analyze(self, sensor_data):

        results = []

        for feature in FEATURES:

            value = sensor_data.get(feature)

            result = self.analyze_sensor(
                feature,
                value
            )

            results.append(result)


        # ----------------------------------------------------
        # Rank abnormal sensors
        # ----------------------------------------------------

        abnormal = [
            result
            for result in results
            if result["status"] in
            ["UNUSUAL", "DEVIATING", "MISSING"]
        ]


        abnormal.sort(
            key=lambda x: (
                x["severity"] != "HIGH",
                -(x["z_score"] or 999)
            )
        )


        # ----------------------------------------------------
        # Overall pattern status
        # ----------------------------------------------------

        missing_count = sum(
            1
            for result in results
            if result["status"] == "MISSING"
        )

        unusual_count = sum(
            1
            for result in results
            if result["status"] == "UNUSUAL"
        )

        deviating_count = sum(
            1
            for result in results
            if result["status"] == "DEVIATING"
        )


        if missing_count > 0:

            overall_status = "SENSOR_DATA_MISSING"

        elif unusual_count > 0:

            overall_status = "UNUSUAL_PATTERN"

        elif deviating_count > 0:

            overall_status = "DEVIATING_PATTERN"

        else:

            overall_status = "NORMAL_PATTERN"


        return {
            "overall_status": overall_status,
            "abnormal_sensors": abnormal,
            "all_sensors": results,
            "abnormal_count": len(abnormal),
            "missing_count": missing_count,
            "unusual_count": unusual_count,
            "deviating_count": deviating_count,
        }


# ============================================================
# TEST FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("SOLARSENTINEL-X SENSOR PATTERN ANALYZER")
    print("=" * 70)

    analyzer = SensorPatternAnalyzer()

    print()
    print("Dataset:")
    print(analyzer.dataset_path)

    print()
    print("Training samples:",
          len(analyzer.dataset))

    print()
    print("Testing analyzer with first dataset row...")

    row = analyzer.dataset.iloc[0]

    sensor_data = {
        feature: row[feature]
        for feature in FEATURES
    }

    result = analyzer.analyze(sensor_data)

    print()
    print("Overall Pattern:",
          result["overall_status"])

    print()
    print("-" * 70)

    for sensor in result["all_sensors"]:

        print(
            f"{sensor['sensor']:25s} | "
            f"Value: {str(sensor['value']):10s} | "
            f"Status: {sensor['status']:10s} | "
            f"Severity: {sensor['severity']}"
        )

    print("-" * 70)

    print()
    print("Abnormal Sensors:",
          result["abnormal_count"])

    print()
    print("Analyzer test completed successfully.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()