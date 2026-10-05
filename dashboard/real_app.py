"""
SolarSentinel-X Real-Time Streamlit Dashboard

Displays live ESP32 sensor data and real AI predictions.

Pipeline:
ESP32
  ↓
Real Digital Twin
  ↓
RF + Autoencoder + LSTM
  ↓
SHAP
  ↓
Decision Engine
  ↓
Streamlit Dashboard
"""

import sys
import time
import math
from pathlib import Path
from collections import deque

import streamlit as st
import pandas as pd
import serial

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
from digital_twin.real.visual_twin import create_visual_twin
from digital_twin.real.what_if_simulator import WhatIfSimulator

from predictive_maintenance.sensor_pattern_analyzer import SensorPatternAnalyzer
from predictive_maintenance.temporal_trend_analyzer import TemporalTrendAnalyzer
from predictive_maintenance.predictive_health_analyzer import PredictiveHealthAnalyzer
from predictive_maintenance.diagnostic_recommendation_engine import DiagnosticRecommendationEngine

# ==========================================================
# CONFIGURATION
# ==========================================================

PORT = "COM15"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 2

HISTORY_LENGTH = 30


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="SolarSentinel-X Real-Time",
    page_icon="☀️",
    layout="wide"
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "serial_connection" not in st.session_state:
    st.session_state.serial_connection = None

if "digital_twin" not in st.session_state:
    st.session_state.digital_twin = None

if "shap_explainer" not in st.session_state:
    st.session_state.shap_explainer = None

if "decision_engine" not in st.session_state:
    st.session_state.decision_engine = None
    
if "pattern_analyzer" not in st.session_state:
    st.session_state.pattern_analyzer = None

if "trend_analyzer" not in st.session_state:
    st.session_state.trend_analyzer = None

if "predictive_health_analyzer" not in st.session_state:
    st.session_state.predictive_health_analyzer = None
    
if "diagnostic_engine" not in st.session_state:
    st.session_state.diagnostic_engine = None

if "latest_diagnostic" not in st.session_state:
    st.session_state.latest_diagnostic = None

if "what_if_simulator" not in st.session_state:
    st.session_state.diagnostic_engine = None
    st.session_state.latest_diagnostic = None
    st.session_state.what_if_simulator = None

if "what_if_result" not in st.session_state:
    st.session_state.what_if_result = None

if "what_if_sensor_data" not in st.session_state:
    st.session_state.what_if_sensor_data = None

if "sensor_history" not in st.session_state:
    st.session_state.sensor_history = deque(
        maxlen=HISTORY_LENGTH
    )

if "latest_state" not in st.session_state:
    st.session_state.latest_state = None

if "latest_shap" not in st.session_state:
    st.session_state.latest_shap = None

if "latest_decision" not in st.session_state:
    st.session_state.latest_decision = None

if "running" not in st.session_state:
    st.session_state.running = False
    
if "sensor_failure" not in st.session_state:
    st.session_state.sensor_failure = None
    st.session_state.last_valid_sensor_data = None

if "last_valid_sensor_data" not in st.session_state:
    st.session_state.last_valid_sensor_data = None


# ==========================================================
# INITIALIZE AI SYSTEM
# ==========================================================

def initialize_system():

    if st.session_state.digital_twin is None:
        st.session_state.digital_twin = (
            RealDigitalTwin()
        )

    if st.session_state.shap_explainer is None:
        st.session_state.shap_explainer = (
            RealSHAPExplainer()
        )

    if st.session_state.decision_engine is None:
        st.session_state.decision_engine = (
            RealDecisionEngine()
        )

    if st.session_state.pattern_analyzer is None:
        st.session_state.pattern_analyzer = (
            SensorPatternAnalyzer()
        )

    if st.session_state.trend_analyzer is None:
        st.session_state.trend_analyzer = (
            TemporalTrendAnalyzer()
        )

    if st.session_state.predictive_health_analyzer is None:
        st.session_state.predictive_health_analyzer = (
            PredictiveHealthAnalyzer()
        )

    if st.session_state.diagnostic_engine is None:
        st.session_state.diagnostic_engine = DiagnosticRecommendationEngine()

    if st.session_state.what_if_simulator is None:
        st.session_state.what_if_simulator = (
            WhatIfSimulator()
        )

# ==========================================================
# SENSOR CONVERSION
# ==========================================================

def convert_sensor_data(parsed_data):

    dht22_temperature = parsed_data["dht22_temp"]
    ds18b20_temperature = parsed_data["ds18b20_temp"]

    try:
        if not math.isfinite(float(ds18b20_temperature)):
            ds18b20_temperature = (
                0.786428 * float(dht22_temperature)
                + 5.217375
            )
    except (TypeError, ValueError):
        ds18b20_temperature = (
            0.786428 * float(dht22_temperature)
            + 5.217375
        )

    return {
        "LDR": parsed_data["ldr"],
        "DHT22_Temperature": dht22_temperature,
        "DHT22_Humidity": parsed_data["dht22_humidity"],
        "DS18B20_Temperature": ds18b20_temperature,
        "Solar_Voltage": parsed_data["ina1_voltage"],
        "Solar_Current": parsed_data["ina1_current"],
        "Solar_Power": parsed_data["ina1_power"],
        "Battery_Voltage": parsed_data["ina2_voltage"],
        "Battery_Current": parsed_data["ina2_current"],
        "Battery_Power": parsed_data["ina2_power"],
    }
    
def validate_sensor_data(parsed_data):

    required_sensors = {
        "LDR": parsed_data["ldr"],
        "DHT22_Temperature": parsed_data["dht22_temp"],
        "DHT22_Humidity": parsed_data["dht22_humidity"],
        "Solar_Voltage": parsed_data["ina1_voltage"],
        "Solar_Current": parsed_data["ina1_current"],
        "Solar_Power": parsed_data["ina1_power"],
        "Battery_Voltage": parsed_data["ina2_voltage"],
        "Battery_Current": parsed_data["ina2_current"],
        "Battery_Power": parsed_data["ina2_power"],
    }

    # Physical plausibility limits for live sensor input.
    # These protect the AI models from corrupted/disconnected INA219
    # readings that may still arrive as numeric values.
    valid_ranges = {
        "LDR": (0, 4095),
        "DHT22_Temperature": (-40, 80),
        "DHT22_Humidity": (0, 100),
        "Solar_Voltage": (0, 15),
        "Solar_Current": (-5, 10),
        "Solar_Power": (0, 100),
        "Battery_Voltage": (0, 15),
        "Battery_Current": (-5, 10),
        "Battery_Power": (0, 100),
    }

    for sensor, value in required_sensors.items():
        try:
            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                return False, sensor

            minimum, maximum = valid_ranges[sensor]

            if numeric_value < minimum or numeric_value > maximum:
                return False, sensor

        except (TypeError, ValueError):
            return False, sensor

    return True, None
    
# ==========================================================
# SERIAL CONNECTION
# ==========================================================

def connect_serial():

    if (
        st.session_state.serial_connection
        is not None
        and st.session_state.serial_connection.is_open
    ):
        return True

    try:

        st.session_state.serial_connection = serial.Serial(
            PORT,
            BAUD_RATE,
            timeout=SERIAL_TIMEOUT
        )

        time.sleep(2)

        return True

    except serial.SerialException as e:

        st.error(
            f"Could not connect to {PORT}: {e}"
        )

        return False


# ==========================================================
# READ ONE SENSOR SAMPLE
# ==========================================================

def read_sensor_sample():

    ser = st.session_state.serial_connection

    if ser is None or not ser.is_open:
        return None

    start_time = time.time()

    while time.time() - start_time < 5:

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

        try:

            parsed = parse_sensor_line(
                line
            )

            valid, failed_sensor = validate_sensor_data(
                parsed
            )

            if not valid:

                st.session_state.sensor_failure = {
                    "sensor": failed_sensor,
                    "reason": "Required Sensor Failure",
                    "status": "INVALID / MISSING"
                }

                return None

            st.session_state.sensor_failure = None

            return convert_sensor_data(
                parsed
            )

        except Exception:

            continue

    return None


# ==========================================================
# PROCESS ONE SAMPLE
# ==========================================================

def process_sample(sensor_data):

    twin = st.session_state.digital_twin

    shap_explainer = (
        st.session_state.shap_explainer
    )

    decision_engine = (
        st.session_state.decision_engine
    )

    pattern_analyzer = (
        st.session_state.pattern_analyzer
    )

    trend_analyzer = (
        st.session_state.trend_analyzer
    )

    predictive_health_analyzer = (
        st.session_state.predictive_health_analyzer
    )

    # ----------------------------------------------
    # Digital Twin
    # ----------------------------------------------

    state = twin.update(
        sensor_data
    )

    # ----------------------------------------------
    # SHAP
    # ----------------------------------------------

    shap_result = (
        shap_explainer.explain(
            sensor_data
        )
    )

    # ----------------------------------------------
    # Decision Engine
    # ----------------------------------------------

    rf_health = state[
        "random_forest"
    ]["prediction"]

    anomaly_status = state[
        "autoencoder"
    ]["status"]

    lstm_health = state[
        "lstm"
    ]["prediction"]

    decision = decision_engine.evaluate(
        rf_health=rf_health,
        anomaly_status=anomaly_status,
        lstm_health=lstm_health
    )

    # ----------------------------------------------
    # Sensor Pattern Analysis
    # ----------------------------------------------

    sensor_pattern = (
        pattern_analyzer.analyze(
            sensor_data
        )
    )

    # ----------------------------------------------
    # Temporal Trend Analysis
    # ----------------------------------------------

    trend_analyzer.update(
        sensor_data
    )

    trend_summary = (
        trend_analyzer.summary()
    )

    # ----------------------------------------------
    # Predictive Health Analysis
    # ----------------------------------------------

    predictive_health_result = (
        predictive_health_analyzer.analyze(
            rf_result=state["random_forest"],
            lstm_result=state["lstm"],
            ae_result=state["autoencoder"],
            sensor_pattern=sensor_pattern,
            trend_summary=trend_summary
        )
    )

    # ----------------------------------------------
    # Specific Diagnostic Recommendation
    # ----------------------------------------------

    diagnostic_result = st.session_state.diagnostic_engine.analyze(
        rf_result=state["random_forest"],
        lstm_result=state["lstm"],
        ae_result=state["autoencoder"],
        sensor_pattern=sensor_pattern,
        trend_summary=trend_summary
    )

    st.session_state.latest_diagnostic = diagnostic_result

    # ----------------------------------------------
    # Update Digital Twin State
    # ----------------------------------------------

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

    # ----------------------------------------------
    # Get FINAL Digital Twin State
    # ----------------------------------------------

    state = (
        twin.state.get_state()
    )

    # ----------------------------------------------
    # History
    # ----------------------------------------------

    history_row = sensor_data.copy()

    history_row["RF_Health"] = (
        rf_health
    )

    history_row["RF_Confidence"] = (
        state["random_forest"]["confidence"]
    )

    history_row["Anomaly_Status"] = (
        anomaly_status
    )

    history_row["Reconstruction_Error"] = (
        state["autoencoder"][
            "reconstruction_error"
        ]
    )

    history_row["LSTM_Health"] = (
        lstm_health
    )

    history_row["Overall_Status"] = (
        decision["overall_status"]
    )

    if predictive_health_result is not None:

        history_row["Predictive_Health"] = (
            predictive_health_result[
                "predictive_health"
            ]
        )

    st.session_state.sensor_history.append(
        history_row
    )

    # ----------------------------------------------
    # Store Latest Results
    # ----------------------------------------------

    st.session_state.latest_state = (
        state
    )

    st.session_state.latest_shap = (
        shap_result
    )

    st.session_state.latest_decision = (
        decision
    )

# ==========================================================
# HEADER
# ==========================================================

st.title(
    "☀️ SolarSentinel-X"
)

st.subheader(
    "Real-Time Explainable AI Solar Microgrid Health Management"
)

st.caption(
    f"ESP32 Serial Input: {PORT} @ {BAUD_RATE} baud"
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header(
    "System Control"
)

if st.session_state.running:

    if st.sidebar.button(
        "⏹ Stop Live Monitoring"
    ):

        st.session_state.running = False

else:

    if st.sidebar.button(
        "▶ Start Live Monitoring"
    ):

        initialize_system()

        if connect_serial():

            st.session_state.running = True

            st.rerun()


if st.sidebar.button(
    "🔄 Reset Session"
):

    st.session_state.sensor_history.clear()

    st.session_state.latest_state = None

    st.session_state.latest_shap = None

    st.session_state.latest_decision = None

    st.session_state.digital_twin = None

    st.session_state.shap_explainer = None

    st.session_state.decision_engine = None
    st.session_state.pattern_analyzer = None

    st.session_state.trend_analyzer = None

    st.session_state.predictive_health_analyzer = None
    
    st.session_state.what_if_simulator = None
    st.session_state.what_if_result = None
    st.session_state.what_if_sensor_data = None
    st.session_state.sensor_failure = None

    st.rerun()


st.sidebar.markdown("---")

st.sidebar.write(
    "**Real Sensor Features:** 10"
)

st.sidebar.write(
    "**RF:** Real trained model"
)

st.sidebar.write(
    "**Autoencoder:** Real trained model"
)

st.sidebar.write(
    "**LSTM:** Real trained model"
)

st.sidebar.write(
    "**SHAP:** Real-time explanation"
)


# ==========================================================
# INITIAL STATUS
# ==========================================================

if not st.session_state.running:

    st.info(
        "Click **Start Live Monitoring** to begin receiving "
        "real ESP32 sensor data."
    )

    st.stop()


    # ==========================================================
# LIVE SAMPLE
# ==========================================================

initialize_system()

sensor_data = read_sensor_sample()

if sensor_data is not None:

    st.session_state.last_valid_sensor_data = sensor_data

    process_sample(
        sensor_data
    )

else:

    if st.session_state.sensor_failure is None:

        st.warning(
            "Waiting for valid ESP32 sensor data..."
        )


# ==========================================================
# CURRENT SYSTEM STATUS
# ==========================================================

state = st.session_state.latest_state

decision = st.session_state.latest_decision

twin_state = None

if st.session_state.digital_twin is not None:
    twin_state = (
        st.session_state.digital_twin
        .state
        .get_state()
    )

# ==========================================================
# SENSOR FAILURE MODE
# ==========================================================

if st.session_state.sensor_failure is not None:

    failure = st.session_state.sensor_failure

    st.markdown("---")
    st.error("🔴 CRITICAL — SENSOR FAILURE")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Reason:**")
        st.write(failure["reason"])

        st.write("**Affected Sensor:**")
        st.write(failure["sensor"])

        st.write("**Data Status:**")
        st.write(failure["status"])

    with col2:
        st.write("**AI Model Status:**")
        st.write("Inference unavailable")

        st.write("**Digital Twin:**")
        st.write("🔴 CRITICAL")

        st.write("**Decision Engine:**")
        st.write("IMMEDIATE INSPECTION REQUIRED")

    st.warning(
        "Live AI results are hidden while the current sensor sample "
        "is invalid. Reconnect the sensor to resume normal inference."
    )

    failure_twin_sensor_data = st.session_state.last_valid_sensor_data

    # If the failure happens before the first valid sample, use a
    # visualization-only baseline. No AI inference is performed.
    if failure_twin_sensor_data is None:
        failure_twin_sensor_data = {
            "LDR": 0.0,
            "DHT22_Temperature": 0.0,
            "DHT22_Humidity": 0.0,
            "DS18B20_Temperature": 0.0,
            "Solar_Voltage": 0.0,
            "Solar_Current": 0.0,
            "Solar_Power": 0.0,
            "Battery_Voltage": 0.0,
            "Battery_Current": 0.0,
            "Battery_Power": 0.0,
        }

    st.markdown("---")
    st.header("🪞 Real-Time 3D Digital Twin")

    twin_fig = create_visual_twin(
        sensor_data=failure_twin_sensor_data,
        overall_status="CRITICAL"
    )

    st.plotly_chart(
        twin_fig,
        use_container_width=True,
        key="sensor_failure_digital_twin"
    )

    st.error(
        "🔴 Digital Twin: CRITICAL — Sensor/data integrity failure."
    )

    st.caption(
        "Visualization-only failure state. AI inference is unavailable "
        "until valid sensor data is restored."
    )

    # Stop this render before any stale AI sections below.
    # The rerun keeps checking for sensor recovery.
    time.sleep(2)
    st.rerun()

if state is not None:

    st.markdown("---")

    st.header(
        "🔴 Live System Status"
    )

    # ------------------------------------------------------
    # STATUS CARDS
    # ------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    rf = state[
        "random_forest"
    ]

    ae = state[
        "autoencoder"
    ]

    lstm = state[
        "lstm"
    ]

    with col1:

        st.metric(
            "RF Health",
            rf["prediction"]
        )

        st.caption(
            f"Confidence: "
            f"{rf['confidence']:.2%}"
        )

    with col2:

        st.metric(
            "Autoencoder",
            ae["status"]
        )

        st.caption(
            f"Error: "
            f"{ae['reconstruction_error']:.4f}"
        )

    with col3:

        if lstm["prediction"] is None:

            st.metric(
                "LSTM Health",
                "Waiting"
            )

        else:

            st.metric(
                "LSTM Health",
                lstm["prediction"]
            )

            st.caption(
                f"Confidence: "
                f"{lstm['confidence']:.2%}"
            )

    with col4:

        st.metric(
            "Overall Status",
            decision["overall_status"]
        )


# ==========================================================
# SENSOR DATA
# ==========================================================

if state is not None:

    st.markdown("---")

    st.header(
        "📡 Real-Time Sensor Measurements"
    )

    sensor_df = pd.DataFrame({
        "Sensor": list(
            state["sensor_data"].keys()
        ),
        "Value": list(
            state["sensor_data"].values()
        )
    })

    st.dataframe(
        sensor_df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# AI MODEL DETAILS
# ==========================================================

if state is not None:

    st.markdown("---")

    st.header(
        "🤖 AI Model Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Random Forest"
        )

        probabilities = rf[
            "probabilities"
        ]

        probability_df = pd.DataFrame(
            {
                "Health Class": list(
                    probabilities.keys()
                ),
                "Probability": list(
                    probabilities.values()
                )
            }
        )

        probability_df[
            "Probability"
        ] = probability_df[
            "Probability"
        ].map(
            lambda x: f"{x:.2%}"
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        st.subheader(
            "Autoencoder"
        )

        st.write(
            f"Status: **{ae['status']}**"
        )

        st.write(
            f"Reconstruction Error: "
            f"`{ae['reconstruction_error']:.6f}`"
        )

        st.write(
            f"Threshold: "
            f"`{ae['threshold']:.6f}`"
        )

# ==========================================================
# AI MODEL DETAILS
# ==========================================================

if state is not None:

    st.markdown("---")

    st.header(
        "🤖 AI Model Analysis"
    )

    # ------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------

    st.subheader(
        "🌲 Random Forest"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Health",
            rf["prediction"]
        )

    with col2:

        st.metric(
            "Confidence",
            f"{rf['confidence']:.2%}"
        )

    probabilities = rf[
        "probabilities"
    ]

    probability_df = pd.DataFrame(
        {
            "Health Class": list(
                probabilities.keys()
            ),
            "Probability": list(
                probabilities.values()
            )
        }
    )

    st.dataframe(
        probability_df.style.format(
            {
                "Probability": "{:.2%}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------------
    # AUTOENCODER
    # ------------------------------------------------------

    st.subheader(
        "🔍 Autoencoder — Unknown Fault Detection"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Status",
            ae["status"]
        )

    with col2:

        st.metric(
            "Reconstruction Error",
            f"{ae['reconstruction_error']:.6f}"
        )

    with col3:

        st.metric(
            "Anomaly Threshold",
            f"{ae['threshold']:.6f}"
        )

    # ------------------------------------------------------
    # LSTM
    # ------------------------------------------------------

    st.subheader(
        "🧠 LSTM — Temporal Health Prediction"
    )

    if lstm["prediction"] is None:

        st.warning(
            "LSTM is waiting for 10 consecutive sensor samples."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Temporal Health",
                lstm["prediction"]
            )

        with col2:

            st.metric(
                "LSTM Confidence",
                f"{lstm['confidence']:.2%}"
            )

        if lstm["probabilities"]:

            lstm_probability_df = pd.DataFrame(
                {
                    "Prediction Class": list(
                        lstm["probabilities"].keys()
                    ),
                    "Probability": list(
                        lstm["probabilities"].values()
                    )
                }
            )

            st.dataframe(
                lstm_probability_df.style.format(
                    {
                        "Probability": "{:.2%}"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

# ==========================================================
# 3D DIGITAL TWIN
# ==========================================================

if state is not None:

    st.markdown("---")

    st.header("🪞 Real-Time 3D Digital Twin")

    twin_sensor_data = state["sensor_data"]

    twin_status = state["overall_status"]

    twin_fig = create_visual_twin(
        sensor_data=twin_sensor_data,
        overall_status=twin_status
    )

    st.plotly_chart(
        twin_fig,
        use_container_width=True,
        key="real_time_digital_twin"
    )

    st.caption(
        "The Digital Twin synchronizes the physical solar "
        "microgrid state with live ESP32 sensor measurements "
        "and AI-derived system health."
    )

# ==========================================================
# DIGITAL TWIN STATE
# ==========================================================

if state is not None:

    st.markdown("---")

    st.header(
        "🪞 Digital Twin State"
    )

    twin_state = (
        st.session_state.digital_twin
        .state
        .get_state()
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Digital Twin Status",
            twin_state["overall_status"]
            if twin_state["overall_status"]
            else "ACTIVE"
        )

    with col2:

        st.metric(
            "RF State",
            twin_state[
                "random_forest"
            ]["prediction"]
        )

    with col3:

        st.metric(
            "LSTM State",
            twin_state[
                "lstm"
            ]["prediction"]
            if twin_state[
                "lstm"
            ]["prediction"]
            else "Waiting"
        )

    st.subheader(
        "Current Twin Sensor State"
    )

    twin_sensor_df = pd.DataFrame(
        {
            "Parameter": list(
                twin_state["sensor_data"].keys()
            ),
            "Current Value": list(
                twin_state["sensor_data"].values()
            )
        }
    )

    st.dataframe(
        twin_sensor_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Twin AI State"
    )

    twin_ai_df = pd.DataFrame(
        [
            {
                "Model": "Random Forest",
                "Result": twin_state[
                    "random_forest"
                ]["prediction"],
                "Confidence": twin_state[
                    "random_forest"
                ]["confidence"]
            },
            {
                "Model": "Autoencoder",
                "Result": twin_state[
                    "autoencoder"
                ]["status"],
                "Confidence": None
            },
            {
                "Model": "LSTM",
                "Result": twin_state[
                    "lstm"
                ]["prediction"]
                if twin_state[
                    "lstm"
                ]["prediction"]
                else "Waiting",
                "Confidence": twin_state[
                    "lstm"
                ]["confidence"]
            }
        ]
    )

    st.dataframe(
        twin_ai_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Twin Decision State"
    )

    st.write(
        f"**Overall Status:** "
        f"{twin_state['overall_status']}"
    )

    st.write(
        f"**Recommendation:** "
        f"{twin_state['recommendation']}"
    )

    st.write(
        f"**Alert:** "
        f"{twin_state['alert']}"
    )

# ==========================================================
# PREDICTIVE HEALTH STATE
# ==========================================================

if twin_state is not None:

    st.subheader(
        "🔮 Predictive Health State"
    )

    predictive_health = twin_state.get(
        "predictive_health"
    )

    predictive_evidence = twin_state.get(
        "predictive_evidence_level"
    )

    predictive_indicators = twin_state.get(
        "predictive_indicators",
        []
    )

    predictive_recommendation = twin_state.get(
        "predictive_recommendation"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predictive Health",
            predictive_health
            if predictive_health
            else "Waiting"
        )

    with col2:

        st.metric(
            "Evidence Level",
            predictive_evidence
            if predictive_evidence
            else "Waiting"
        )

    if predictive_indicators:

        st.write("**Predictive Health Indicators:**")

        for indicator in predictive_indicators:

            st.write(
                f"• {indicator}"
            )

    if predictive_recommendation:

        st.info(
            f"**Predictive Recommendation:** "
            f"{predictive_recommendation}"
        )

# ==========================================================
# SPECIFIC DIAGNOSTIC RECOMMENDATION
# ==========================================================

diagnostic = st.session_state.get("latest_diagnostic")

if diagnostic:
    st.subheader("🩺 Specific Diagnostic Recommendation")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Diagnostic Status", diagnostic.get("diagnostic_status", "UNKNOWN"))
    with col2:
        st.metric("Evidence Strength", diagnostic.get("evidence_strength", "LOW"))

    st.markdown("### 🔍 Affected Subsystems")
    affected = diagnostic.get("affected_subsystems", [])
    if affected:
        for subsystem in affected:
            st.write(f"• **{subsystem}**")
    else:
        st.info("No specific subsystem requires diagnostic attention.")

    st.markdown("### 📊 Sensor-Level Diagnostics")
    for item in diagnostic.get("sensor_diagnostics", []):
        st.markdown(f"#### 🔹 {item.get('sensor', 'Unknown')}")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.write("**Value**")
            st.write(item.get("value", "N/A"))
        with c2:
            st.write("**Pattern**")
            st.write(item.get("status", "N/A"))
        with c3:
            st.write("**Deviation**")
            d = item.get("deviation_percent")
            st.write(f"{d:.2f}%" if d is not None else "N/A")
        with c4:
            st.write("**Recent Trend**")
            st.write(item.get("trend", "N/A"))
        st.write(item.get("message", ""))
        st.caption(
            f"Subsystem: {item.get('subsystem', 'N/A')} | "
            f"Component: {item.get('component', 'N/A')} | "
            f"Evidence: {item.get('evidence_strength', 'N/A')}"
        )

    st.markdown("### 🛠️ Recommended Action")
    st.info(diagnostic.get("recommendation", "Continue monitoring."))

    recommendations = diagnostic.get("subsystem_recommendations", {})
    if recommendations:
        st.markdown("### 🎯 Subsystem-Specific Guidance")
        for subsystem, recommendation in recommendations.items():
            st.write(f"**{subsystem}:** {recommendation}")

    points = diagnostic.get("inspection_points", [])
    if points:
        st.markdown("### 🔧 Suggested Verification Points")
        for point in points:
            st.write(f"• {point}")

# ==========================================================
# SHAP EXPLANATION
# ==========================================================

if st.session_state.latest_shap is not None:

    st.markdown("---")

    st.header(
        "🔎 Live SHAP Explanation"
    )

    shap_result = (
        st.session_state.latest_shap
    )

    st.write(
        f"Prediction explained: "
        f"**{shap_result['prediction']}**"
    )

    shap_df = pd.DataFrame(
        shap_result["top_features"]
    )

    shap_df = shap_df.rename(
        columns={
            "feature": "Feature",
            "shap_value": "SHAP Value",
            "absolute_shap": "Absolute SHAP",
            "influence": "Influence"
        }
    )

    st.dataframe(
        shap_df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# DECISION ENGINE
# ==========================================================

if decision is not None:

    st.markdown("---")

    st.header(
        "🧠 Decision Engine"
    )

    st.subheader(
        decision["overall_status"]
    )

    st.write(
        "**Recommendation:**"
    )

    st.info(
        decision["recommendation"]
    )

    st.write(
        "**Alert:**"
    )

    if decision["overall_status"] == "CRITICAL":

        st.error(
            decision["alert"]
        )

    elif decision["overall_status"] == "WARNING":

        st.warning(
            decision["alert"]
        )

    else:

        st.success(
            decision["alert"]
        )


# ==========================================================
# SENSOR HISTORY
# ==========================================================

if len(
    st.session_state.sensor_history
) > 1:

    st.markdown("---")

    st.header(
        "📈 Live Sensor History"
    )

    history_df = pd.DataFrame(
        list(
            st.session_state.sensor_history
        )
    )

    st.line_chart(
        history_df[
            [
                "LDR",
                "DHT22_Temperature",
                "DHT22_Humidity",
                "DS18B20_Temperature"
            ]
        ]
    )

    st.subheader(
        "Solar Measurements"
    )

    st.line_chart(
        history_df[
            [
                "Solar_Voltage",
                "Solar_Current",
                "Solar_Power"
            ]
        ]
    )

    st.subheader(
        "Battery Measurements"
    )

    st.line_chart(
        history_df[
            [
                "Battery_Voltage",
                "Battery_Current",
                "Battery_Power"
            ]
        ]
    )

# ==========================================================
# WHAT-IF DIGITAL TWIN SIMULATION
# ==========================================================

if state is not None:

    st.markdown("---")

    st.header("🔮 Digital Twin — What-If Simulation")

    st.info(
        "SIMULATION MODE: Adjust hypothetical sensor conditions "
        "and evaluate the response of the trained AI models. "
        "These values do not replace the live ESP32 measurements."
    )

    simulator = st.session_state.what_if_simulator

    # ======================================================
    # IMPORTANT: WHAT-IF INPUTS ARE INDEPENDENT OF ESP32
    # ======================================================
    # These are hypothetical starting values. They are NOT copied
    # from the live ESP32 measurements. User can change them freely.
    what_if_defaults = {
        "what_if_ldr": 2000.0,
        "what_if_dht_temp": 30.0,
        "what_if_humidity": 60.0,
        "what_if_ds_temp": 30.0,
        "what_if_solar_voltage": 5.0,
        "what_if_solar_current": 1.0,
        "what_if_solar_power": 5.0,
        "what_if_battery_voltage": 7.0,
        "what_if_battery_current": 0.5,
        "what_if_battery_power": 3.0,
    }

    for key, default_value in what_if_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # ======================================================
    # WHAT-IF INPUT FORM
    # ======================================================

    st.info("🧪 Independent Simulation Inputs — these values are hypothetical and are NOT taken from ESP32.")

    with st.form("what_if_simulation_form"):

        st.subheader("⚙️ Configure Hypothetical Operating Condition")

        col1, col2 = st.columns(2)

        with col1:

            simulated_ldr = st.slider(
                "LDR",
                min_value=0.0,
                max_value=4095.0,
                step=1.0,
                key="what_if_ldr"
            )

            simulated_dht_temp = st.slider(
                "DHT22 Temperature (°C)",
                min_value=15.0,
                max_value=60.0,
                step=0.1,
                key="what_if_dht_temp"
            )

            simulated_dht_humidity = st.slider(
                "DHT22 Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="what_if_humidity"
            )

            simulated_ds18b20 = st.slider(
                "DS18B20 Temperature (°C)",
                min_value=15.0,
                max_value=60.0,
                step=0.1,
                key="what_if_ds_temp"
            )

            simulated_solar_voltage = st.slider(
                "Solar Voltage (V)",
                min_value=0.0,
                max_value=10.0,
                step=0.01,
                key="what_if_solar_voltage"
            )

        with col2:

            simulated_solar_current = st.slider(
                "Solar Current (A)",
                min_value=-2.0,
                max_value=10.0,
                step=0.01,
                key="what_if_solar_current"
            )

            simulated_solar_power = st.slider(
                "Solar Power (W)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                key="what_if_solar_power"
            )

            simulated_battery_voltage = st.slider(
                "Battery Voltage (V)",
                min_value=0.0,
                max_value=15.0,
                step=0.01,
                key="what_if_battery_voltage"
            )

            simulated_battery_current = st.slider(
                "Battery Current (A)",
                min_value=-5.0,
                max_value=10.0,
                step=0.01,
                key="what_if_battery_current"
            )

            simulated_battery_power = st.slider(
                "Battery Power (W)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                key="what_if_battery_power"
            )

        # ==================================================
        # SIMULATION BUTTON
        # ==================================================

        run_simulation = st.form_submit_button(
            "🔮 RUN WHAT-IF SIMULATION",
            use_container_width=True
        )

    # ======================================================
    # RUN SIMULATION ONLY WHEN BUTTON IS PRESSED
    # ======================================================

    if run_simulation:

        simulated_sensor_data = {

            "LDR": simulated_ldr,

            "DHT22_Temperature":
                simulated_dht_temp,

            "DHT22_Humidity":
                simulated_dht_humidity,

            "DS18B20_Temperature":
                simulated_ds18b20,

            "Solar_Voltage":
                simulated_solar_voltage,

            "Solar_Current":
                simulated_solar_current,

            "Solar_Power":
                simulated_solar_power,

            "Battery_Voltage":
                simulated_battery_voltage,

            "Battery_Current":
                simulated_battery_current,

            "Battery_Power":
                simulated_battery_power
        }

        # Run the real trained models
        simulation_result = simulator.simulate(
            simulated_sensor_data
        )

        # Store permanently in session state
        st.session_state.what_if_result = simulation_result

        st.session_state.what_if_sensor_data = (
            simulated_sensor_data
        )

    # ======================================================
    # DISPLAY STORED SIMULATION RESULT
    # ======================================================

    if st.session_state.what_if_result is not None:

        simulation_result = st.session_state.what_if_result

        simulated_sensor_data = (
            st.session_state.what_if_sensor_data
        )

        # Actual WhatIfSimulator return structure
        rf_result = simulation_result["random_forest"]
        ae_result = simulation_result["autoencoder"]
        decision_result = simulation_result["decision"]

        st.markdown("---")
        st.subheader("🔮 What-If Simulation Result")

        st.warning(
            "SIMULATION ONLY — This result is generated from "
            "hypothetical sensor values and does not represent "
            "the current ESP32 measurements."
        )

        # ==================================================
        # MODEL RESULTS
        # ==================================================

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "🌲 Simulated RF Health",
                rf_result["health"]
            )

            st.metric(
                "RF Confidence",
                f"{rf_result['confidence']:.2f}%"
            )

        with result_col2:

            st.metric(
                "🔍 Simulated Autoencoder",
                ae_result["status"]
            )

            st.metric(
                "Reconstruction Error",
                f"{ae_result['reconstruction_error']:.6f}"
            )

        with result_col3:

            st.metric(
                "🧠 Simulated Twin State",
                decision_result["overall_status"]
            )

            st.write(
                decision_result["recommendation"]
            )

        # ==================================================
        # RF PROBABILITIES
        # ==================================================

        st.subheader("🌲 Simulated Random Forest Probabilities")

        rf_probabilities = rf_result["probabilities"]

        prob_col1, prob_col2, prob_col3 = st.columns(3)

        with prob_col1:

            st.metric(
                "Healthy",
                f"{rf_probabilities.get('Healthy', 0):.2f}%"
            )

        with prob_col2:

            st.metric(
                "Warning",
                f"{rf_probabilities.get('Warning', 0):.2f}%"
            )

        with prob_col3:

            st.metric(
                "Critical",
                f"{rf_probabilities.get('Critical', 0):.2f}%"
            )

        # ==================================================
        # AUTOENCODER
        # ==================================================

        st.subheader("🔍 Simulated Autoencoder")

        ae_col1, ae_col2, ae_col3 = st.columns(3)

        with ae_col1:
            st.write("Status")
            st.write(f"**{ae_result['status']}**")

        with ae_col2:
            st.write("Reconstruction Error")
            st.write(
                f"`{ae_result['reconstruction_error']:.6f}`"
            )

        with ae_col3:
            st.write("Threshold")
            st.write(
                f"`{ae_result['threshold']:.6f}`"
            )

        # ==================================================
        # LSTM
        # ==================================================

        st.subheader("🧠 Simulated LSTM")

        st.info(
            "LSTM was not evaluated for this single what-if "
            "scenario because the real LSTM requires a "
            "10-sample temporal sequence."
        )

        # ==================================================
        # SIMULATED DECISION ENGINE
        # ==================================================

        st.subheader("🧠 Simulated Decision Engine")

        decision_col1, decision_col2 = st.columns(2)

        with decision_col1:

            st.write("Overall Status")

            if decision_result["overall_status"] == "CRITICAL":

                st.error("🔴 CRITICAL")

            elif decision_result["overall_status"] == "WARNING":

                st.warning("🟠 WARNING")

            else:

                st.success("🟢 NORMAL")

        with decision_col2:

            st.write("Recommendation")

            st.write(
                decision_result["recommendation"]
            )

        if decision_result.get("alert"):

            st.warning(
                decision_result["alert"]
            )

        # ==================================================
        # SIMULATED SENSOR VALUES
        # ==================================================

        st.subheader("📡 Simulated Sensor State")

        simulated_df = pd.DataFrame(
            [simulated_sensor_data]
        )

        st.dataframe(
            simulated_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "🔮 SIMULATION ONLY — The physical ESP32 system "
            "continues operating independently."
        )



# ==========================================================
# AUTO REFRESH
# ==========================================================

if st.session_state.running:
    time.sleep(10)
    st.rerun()
