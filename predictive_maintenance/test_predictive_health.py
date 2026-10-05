# ============================================================
# PROJECT ROOT PATH
# ============================================================

import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from models.real_inference import RealModelInference

from predictive_maintenance.sensor_pattern_analyzer import (
    SensorPatternAnalyzer
)

from predictive_maintenance.temporal_trend_analyzer import (
    TemporalTrendAnalyzer
)

from predictive_maintenance.predictive_health_analyzer import (
    PredictiveHealthAnalyzer
)

# ============================================================
# TEST SENSOR DATA
# ============================================================

sensor_data = {
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


# ============================================================
# INITIALIZE COMPONENTS
# ============================================================

print("\n" + "=" * 70)
print("SOLARSENTINEL-X - STEP 3A INTEGRATION TEST")
print("=" * 70)

print("\nInitializing real model inference...")

inference = RealModelInference()

pattern_analyzer = SensorPatternAnalyzer()

trend_analyzer = TemporalTrendAnalyzer()

predictive_health = PredictiveHealthAnalyzer()


# ============================================================
# FEED 10 REAL-FORMAT OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("FEEDING SENSOR OBSERVATIONS")
print("=" * 70)

result = None

for sample_number in range(1, 11):

    print(
        f"\nProcessing sample "
        f"{sample_number}/10..."
    )

    # --------------------------------------------------------
    # REAL AI MODELS
    # --------------------------------------------------------

    result = inference.predict(sensor_data)

    # --------------------------------------------------------
    # SENSOR PATTERN ANALYSIS
    # --------------------------------------------------------

    pattern_result = pattern_analyzer.analyze(
        sensor_data
    )

    # --------------------------------------------------------
    # TEMPORAL TREND ANALYSIS
    # --------------------------------------------------------

    trend_result = trend_analyzer.update(
        sensor_data
    )


# ============================================================
# EXTRACT REAL MODEL RESULTS
# ============================================================

rf_result = result["random_forest"]

ae_result = result["autoencoder"]

lstm_result = result["lstm"]


# ============================================================
# TREND SUMMARY
# ============================================================

trend_summary = trend_analyzer.summary()


# ============================================================
# PREDICTIVE HEALTH ANALYSIS
# ============================================================

health_result = predictive_health.analyze(
    rf_result=rf_result,
    lstm_result=lstm_result,
    ae_result=ae_result,
    sensor_pattern=pattern_result,
    trend_summary=trend_summary,
)


# ============================================================
# DISPLAY REAL MODEL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("REAL MODEL RESULTS")
print("=" * 70)

print(
    "\nRandom Forest:"
)

print(
    "Prediction :",
    rf_result["prediction"]
)

print(
    "Confidence :",
    round(
        rf_result["confidence"],
        4
    )
)


print(
    "\nAutoencoder:"
)

print(
    "Status :",
    ae_result["status"]
)

print(
    "Anomaly :",
    ae_result["is_anomaly"]
)

print(
    "Reconstruction Error :",
    round(
        ae_result["reconstruction_error"],
        6
    )
)


print(
    "\nLSTM:"
)

print(
    "Status :",
    lstm_result["status"]
)

if lstm_result["ready"]:

    print(
        "Prediction :",
        lstm_result["prediction"]
    )

    print(
        "Confidence :",
        round(
            lstm_result["confidence"],
            4
        )
    )

else:

    print(
        "Waiting :",
        f"{lstm_result['samples_collected']}/"
        f"{lstm_result['samples_required']}"
    )


# ============================================================
# DISPLAY SENSOR PATTERN
# ============================================================

print("\n" + "=" * 70)
print("SENSOR PATTERN ANALYSIS")
print("=" * 70)

print(
    "Overall Status :",
    pattern_result["overall_status"]
)

print(
    "Abnormal Sensors :",
    pattern_result["abnormal_count"]
)


# ============================================================
# DISPLAY TEMPORAL TREND
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL TREND ANALYSIS")
print("=" * 70)

print("Trend Summary:")

for key, value in trend_summary.items():

    print(
        f"{key} : {value}"
    )

# ============================================================
# DISPLAY PREDICTIVE HEALTH
# ============================================================

print("\n" + "=" * 70)
print("PREDICTIVE HEALTH RESULT")
print("=" * 70)

print(
    "\nPredictive Health :",
    health_result["predictive_health"]
)

print(
    "Evidence Level    :",
    health_result["evidence_level"]
)

print(
    "\nIndicators:"
)

for indicator in health_result["indicators"]:

    print(
        " -",
        indicator
    )


print("\nRecommendations:")

recommendations = (
    health_result.get("recommendations")
    or health_result.get("recommendation")
    or []
)

if isinstance(recommendations, str):
    recommendations = [recommendations]

for recommendation in recommendations:
    print(" -", recommendation)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 3A TEST COMPLETED")
print("=" * 70)