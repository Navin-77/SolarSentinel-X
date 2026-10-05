# ============================================================
# SolarSentinel-X
# Predictive Maintenance — Temporal Trend Analyzer
# ============================================================

from collections import deque
import numpy as np


class TemporalTrendAnalyzer:

    # --------------------------------------------------------
    # REAL SENSOR FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def __init__(self, window_size=10):

        self.window_size = window_size

        self.history = {
            feature: deque(maxlen=window_size)
            for feature in self.FEATURES
        }

    # --------------------------------------------------------
    # ADD SENSOR READING
    # --------------------------------------------------------

    def update(self, sensor_data):

        for feature in self.FEATURES:

            value = sensor_data.get(feature)

            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(value):
                continue

            self.history[feature].append(value)

    # --------------------------------------------------------
    # ANALYZE ONE SENSOR
    # --------------------------------------------------------

    def analyze_sensor(self, feature):

        values = list(self.history[feature])

        if len(values) < 2:
            return {
                "sensor": feature,
                "status": "INSUFFICIENT_DATA",
                "trend": "UNKNOWN",
                "current_value": values[-1] if values else None,
                "change": None,
                "change_percent": None,
                "average": None,
            }

        current = values[-1]
        previous = values[-2]

        change = current - previous

        if previous != 0:
            change_percent = (change / abs(previous)) * 100
        else:
            change_percent = None

        average = float(np.mean(values))

        # ----------------------------------------------------
        # TREND CLASSIFICATION
        # ----------------------------------------------------

        if abs(change) < 1e-6:
            trend = "STABLE"

        elif change > 0:
            trend = "INCREASING"

        else:
            trend = "DECREASING"

        return {
            "sensor": feature,
            "status": "AVAILABLE",
            "trend": trend,
            "current_value": current,
            "change": change,
            "change_percent": change_percent,
            "average": average,
            "samples": len(values),
        }

    # --------------------------------------------------------
    # ANALYZE ALL SENSORS
    # --------------------------------------------------------

    def analyze(self):

        results = {}

        for feature in self.FEATURES:
            results[feature] = self.analyze_sensor(feature)

        return results

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def summary(self):

        results = self.analyze()

        increasing = []
        decreasing = []
        stable = []
        insufficient = []

        for feature, result in results.items():

            trend = result["trend"]

            if trend == "INCREASING":
                increasing.append(feature)

            elif trend == "DECREASING":
                decreasing.append(feature)

            elif trend == "STABLE":
                stable.append(feature)

            else:
                insufficient.append(feature)

        return {
            "increasing_sensors": increasing,
            "decreasing_sensors": decreasing,
            "stable_sensors": stable,
            "insufficient_data": insufficient,
            "total_sensors": len(self.FEATURES),
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    analyzer = TemporalTrendAnalyzer(window_size=10)

    sample = {
        "LDR": 800,
        "DHT22_Temperature": 30,
        "DHT22_Humidity": 75,
        "DS18B20_Temperature": 29,
        "Solar_Voltage": 4.8,
        "Solar_Current": 8,
        "Solar_Power": 38,
        "Battery_Voltage": 8,
        "Battery_Current": 1,
        "Battery_Power": 8,
    }

    # Add a few readings
    for i in range(5):

        reading = sample.copy()

        reading["DHT22_Temperature"] += i * 0.2
        reading["Solar_Power"] -= i * 2

        analyzer.update(reading)

    print("\n" + "=" * 60)
    print("TEMPORAL TREND ANALYZER TEST")
    print("=" * 60)

    print("\nSensor Trends:")

    results = analyzer.analyze()

    for sensor, result in results.items():

        print(
            f"{sensor:25} → "
            f"{result['trend']}"
        )

    print("\nSummary:")

    print(analyzer.summary())

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)