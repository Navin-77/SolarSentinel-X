"""
SolarSentinel-X Real-Time System Controller

Complete integration of:

ESP32
  ↓
Serial Reader
  ↓
Real Sensor Feature Adapter
  ↓
Real Digital Twin
  ├── Random Forest
  ├── Autoencoder
  └── LSTM
  ↓
Live SHAP
  ↓
Real Decision Engine
  ↓
Final System State
"""

from pathlib import Path
import sys
import time
import serial
import math

# ----------------------------------------------------------
# PROJECT ROOT
# ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------
# PROJECT MODULES
# ----------------------------------------------------------

from realtime.sensor_interface import parse_sensor_line
from digital_twin.real.real_twin import RealDigitalTwin
from explainability.real_shap_live import RealSHAPExplainer
from decision_engine.real_decision_engine import RealDecisionEngine

# Predictive Maintenance
from predictive_maintenance.sensor_pattern_analyzer import SensorPatternAnalyzer
from predictive_maintenance.maintenance_engine import MaintenanceEngine
from predictive_maintenance.temporal_trend_analyzer import TemporalTrendAnalyzer
from predictive_maintenance.predictive_health_analyzer import (
    PredictiveHealthAnalyzer
)

# ==========================================================
# CONFIGURATION
# ==========================================================

PORT = "COM11"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 2


# ==========================================================
# SENSOR DATA ADAPTER
# ==========================================================

def convert_sensor_data(parsed_data):



    dht22_temperature = float(
        parsed_data["dht22_temp"]
    )



    ds18b20_temperature = parsed_data["ds18b20_temp"]

    try:
        ds18b20_temperature = float(
            ds18b20_temperature
        )
    except (TypeError, ValueError):
        ds18b20_temperature = float("nan")


    if not math.isfinite(ds18b20_temperature):



        ds18b20_temperature = (
            0.786428 * dht22_temperature
            + 5.217375
        )


    return {
        "LDR": parsed_data["ldr"],

        "DHT22_Temperature":
            parsed_data["dht22_temp"],

        "DHT22_Humidity":
            parsed_data["dht22_humidity"],

        "DS18B20_Temperature":
            ds18b20_temperature,

        "Solar_Voltage":
            parsed_data["ina1_voltage"],

        "Solar_Current":
            parsed_data["ina1_current"],

        "Solar_Power":
            parsed_data["ina1_power"],

        "Battery_Voltage":
            parsed_data["ina2_voltage"],

        "Battery_Current":
            parsed_data["ina2_current"],

        "Battery_Power":
            parsed_data["ina2_power"],
    }
    
def validate_sensor_data(parsed_data):

    required_sensors = {
        "LDR": parsed_data["ldr"],
        "DHT22_Temperature": parsed_data["dht22_temp"],
        "DHT22_Humidity": parsed_data["dht22_humidity"],
        "DS18B20_Temperature": parsed_data["ds18b20_temp"],
        "Solar_Voltage": parsed_data["ina1_voltage"],
        "Solar_Current": parsed_data["ina1_current"],
        "Solar_Power": parsed_data["ina1_power"],
        "Battery_Voltage": parsed_data["ina2_voltage"],
        "Battery_Current": parsed_data["ina2_current"],
        "Battery_Power": parsed_data["ina2_power"],
    }

    for sensor, value in required_sensors.items():
        try:
            if value is None or not math.isfinite(float(value)):
                return False, sensor
        except (TypeError, ValueError):
            return False, sensor

    return True, None


# ==========================================================
# DISPLAY SENSOR DATA
# ==========================================================

def display_sensor_data(sensor_data):

    print("\n" + "-" * 60)
    print("REAL SENSOR DATA")
    print("-" * 60)

    for key, value in sensor_data.items():

        print(
            f"{key:25s}: {value}"
        )


# ==========================================================
# DISPLAY AI RESULTS
# ==========================================================

def display_ai_results(result):

    rf = result["random_forest"]
    ae = result["autoencoder"]
    lstm = result["lstm"]

    print("\n" + "-" * 60)
    print("AI MODEL RESULTS")
    print("-" * 60)

    print(
        f"Random Forest Health : "
        f"{rf['prediction']}"
    )

    print(
        f"RF Confidence        : "
        f"{rf['confidence']:.4f}"
    )

    print(
        f"Autoencoder Status    : "
        f"{ae['status']}"
    )

    print(
        f"Reconstruction Error  : "
        f"{ae['reconstruction_error']:.6f}"
    )

    print(
        f"Anomaly Threshold     : "
        f"{ae['threshold']:.6f}"
    )

    if lstm["prediction"] is None:

        print(
            "LSTM Health           : "
            "Waiting for 10 samples"
        )

    else:

        print(
            f"LSTM Health           : "
            f"{lstm['prediction']}"
        )

        print(
            f"LSTM Confidence       : "
            f"{lstm['confidence']:.4f}"
        )


# ==========================================================
# DISPLAY SHAP
# ==========================================================

def display_shap(result):

    if result is None:
        return

    print("\n" + "-" * 60)
    print("LIVE SHAP EXPLANATION")
    print("-" * 60)

    print(
        f"Prediction explained : "
        f"{result['prediction']}"
    )

    print("\nTop Influential Features:")

    for item in result["top_features"]:

        print(
            f"{item['feature']:25s} "
            f"{item['shap_value']:+.6f} "
            f"({item['influence']})"
        )

# ==========================================================
# DISPLAY PREDICTIVE MAINTENANCE
# ==========================================================

def display_maintenance(result):

    print("\n" + "-" * 60)
    print("PREDICTIVE MAINTENANCE")
    print("-" * 60)

    print(
        f"Status           : "
        f"{result['status']}"
    )

    print(
        f"Severity         : "
        f"{result['severity']}"
    )

    print(
        f"Affected Sensor  : "
        f"{result['affected_sensor']}"
    )

    print(
        f"Component        : "
        f"{result['affected_component']}"
    )

    print(
        f"Pattern          : "
        f"{result['pattern']}"
    )

    print(
        f"\nRecommendation   : "
        f"{result['maintenance_recommendation']}"
    )

    print("\nSuggested Inspection:")

    for index, item in enumerate(
        result["suggested_inspection"],
        start=1
    ):

        print(
            f"  {index}. {item}"
        )

# ==========================================================
# DISPLAY DECISION
# ==========================================================

def display_decision(decision):

    print("\n" + "-" * 60)
    print("REAL DECISION ENGINE")
    print("-" * 60)

    print(
        f"Overall Status : "
        f"{decision['overall_status']}"
    )

    print(
        f"Recommendation : "
        f"{decision['recommendation']}"
    )

    print(
        f"Alert          : "
        f"{decision['alert']}"
    )


# ==========================================================
# MAIN
# ==========================================================

def run():

    print("=" * 60)
    print("SolarSentinel-X REAL SYSTEM CONTROLLER")
    print("=" * 60)

    print("\nInitializing Real Digital Twin...")

    twin = RealDigitalTwin()

    print("\nInitializing Live SHAP...")

    shap_explainer = RealSHAPExplainer()

    print("\nInitializing Real Decision Engine...")

    decision_engine = RealDecisionEngine()
    
    print("\nInitializing Predictive Maintenance...")

    pattern_analyzer = SensorPatternAnalyzer()

    maintenance_engine = MaintenanceEngine(
        pattern_analyzer
    )
    
    trend_analyzer = TemporalTrendAnalyzer()

    predictive_health_analyzer = PredictiveHealthAnalyzer()

    print("Temporal Trend Analyzer initialized.")
    print("Predictive Health Analyzer initialized.")

    print("\nConnecting to ESP32...")
    print(
        f"Port : {PORT}"
    )
    print(
        f"Baud : {BAUD_RATE}"
    )

    try:

        ser = serial.Serial(
            PORT,
            BAUD_RATE,
            timeout=SERIAL_TIMEOUT
        )

        time.sleep(2)

        print("\nESP32 connected successfully.")
        print("Waiting for real sensor data...")
        print("Press Ctrl+C to stop.\n")

        while True:

            line = (
                ser.readline()
                .decode(
                    "utf-8",
                    errors="ignore"
                )
                .strip()
            )

            if not line:
                continue

            if not line.startswith("DATA,"):
                continue
            
            print(f"RECEIVED DATA LINE: {line}")

            # ------------------------------------------------
            # PARSE ESP32 DATA
            # ------------------------------------------------

            try:

                parsed_data = parse_sensor_line(
                    line
                )

            except Exception as e:

                print(
                    f"\nSensor parsing error: {e}"
                )

                continue
            
            valid, failed_sensor = validate_sensor_data(parsed_data)

            if not valid:

                print("\n" + "=" * 60)
                print("CRITICAL - SENSOR FAILURE")
                print("=" * 60)
                print("Reason          : Required Sensor Failure")
                print(f"Affected Sensor : {failed_sensor}")
                print("Data Status     : INVALID / MISSING")
                print("AI Model Status : Inference unavailable")
                print("Digital Twin    : CRITICAL")
                print("Decision Engine : IMMEDIATE INSPECTION REQUIRED")
                print("=" * 60)

                continue

            # ------------------------------------------------
            # CONVERT TO MODEL FEATURES
            # ------------------------------------------------

            try:

                sensor_data = convert_sensor_data(
                    parsed_data
                )

            except Exception as e:

                print(
                    f"\nSensor feature conversion error: {e}"
                )

                continue
            
            # ------------------------------------------------
            # TEMPORAL TREND ANALYSIS
            # ------------------------------------------------

            try:

                trend_result = trend_analyzer.update(
                    sensor_data
                )

            except Exception as e:

                print(
                    f"\nTemporal Trend error: {e}"
                )

                trend_result = None
            
            # ------------------------------------------------
            # PREDICTIVE MAINTENANCE
            # ------------------------------------------------

            try:

                maintenance_result = maintenance_engine.generate(
                    sensor_data
                )

            except Exception as e:

                print(
                    f"\nPredictive Maintenance error: {e}"
                )

                maintenance_result = {
                    "status": "ERROR",
                    "severity": "UNKNOWN",
                    "affected_sensor": "Unknown",
                    "affected_component": "Unknown",
                    "pattern": "Unavailable",
                    "maintenance_recommendation": (
                        "Predictive maintenance analysis unavailable."
                    ),
                    "suggested_inspection": [],
                    "evidence": [],
                    "additional_abnormal_sensors": [],
                }

            # ------------------------------------------------
            # DIGITAL TWIN
            # ------------------------------------------------

            try:

                twin_result = twin.update(
                    sensor_data
                )

            except Exception as e:

                print(
                    f"\nDigital Twin error: {e}"
                )

                continue

            # ------------------------------------------------
            # SHAP
            # ------------------------------------------------

            try:

                shap_result = (
                    shap_explainer.explain(
                        sensor_data
                    )
                )

            except Exception as e:

                print(
                    f"\nSHAP error: {e}"
                )

                shap_result = None

            # ------------------------------------------------
            # DECISION ENGINE
            # ------------------------------------------------

            try:

                rf_health = twin_result[
                    "random_forest"
                ]["prediction"]

                anomaly_status = twin_result[
                    "autoencoder"
                ]["status"]

                lstm_health = twin_result[
                    "lstm"
                ]["prediction"]

                decision = decision_engine.evaluate(
                    rf_health=rf_health,
                    anomaly_status=anomaly_status,
                    lstm_health=lstm_health
                )

            except Exception as e:

                print(
                    f"\nDecision Engine error: {e}"
                )

                continue
            
            # ------------------------------------------------
            # PREDICTIVE HEALTH ANALYSIS
            # ------------------------------------------------

            try:

                trend_summary = trend_analyzer.summary()

                predictive_health_result = (
                    predictive_health_analyzer.analyze(
                        rf_result=twin_result["random_forest"],
                        lstm_result=twin_result["lstm"],
                        ae_result=twin_result["autoencoder"],
                        sensor_pattern=pattern_analyzer.analyze(
                            sensor_data
                        ),
                        trend_summary=trend_summary
                    )
                )

            except Exception as e:

                print(
                    f"\nPredictive Health error: {e}"
                )

                predictive_health_result = None

            # ------------------------------------------------
            # UPDATE DIGITAL TWIN STATE
            # ------------------------------------------------

            twin.state.update_shap(
                shap_result
            )

            twin.state.update_decision(
                decision["overall_status"],
                decision["recommendation"],
                decision["alert"]
            )
            
            if predictive_health_result is not None:
                twin.state.update_predictive_health(
                    predictive_health_result
                )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display_sensor_data(
                sensor_data
            )

            display_ai_results(
                twin_result
            )

            display_shap(
                shap_result
            )

            display_decision(
                decision
            )

            display_maintenance(
                maintenance_result
            )
            
            # ------------------------------------------------
            # DISPLAY PREDICTIVE HEALTH
            # ------------------------------------------------

            if predictive_health_result is not None:

                print("\n" + "-" * 60)
                print("PREDICTIVE HEALTH ANALYSIS")
                print("-" * 60)

                print(
                    f"Predictive Health : "
                    f"{predictive_health_result['predictive_health']}"
                )

                print(
                    f"Evidence Level    : "
                    f"{predictive_health_result['evidence_level']}"
                )

                print("\nIndicators:")

                for indicator in predictive_health_result["indicators"]:

                    print(
                        f" - {indicator}"
                    )

                print("\nRecommendation:")

                recommendation = (
                    predictive_health_result.get(
                        "recommendation",
                        "Continue monitoring the system."
                    )
                )

                print(
                    f" - {recommendation}"
                )

            print("\n" + "=" * 60)
            print("REAL-TIME CYCLE COMPLETED")
            print("=" * 60)

    except serial.SerialException as e:

        print(
            f"\nSerial connection error: {e}"
        )

        print(
            f"Make sure ESP32 is connected to {PORT}"
        )

        print(
            "Also make sure Arduino Serial Monitor is CLOSED."
        )

    except KeyboardInterrupt:

        print(
            "\n\nReal System Controller stopped by user."
        )

    finally:

        try:
            ser.close()
        except:
            pass


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    run()