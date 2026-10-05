"""
Dashboard Data Loader

Runs the complete AI pipeline and returns the latest predictions.
"""
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from digital_twin.digital_twin import DigitalTwin
from decision_engine.decision_engine import DecisionEngine

from datetime import datetime

@st.cache_resource
def get_digital_twin():

    return DigitalTwin()


@st.cache_resource
def get_decision_engine():

    return DecisionEngine()

@st.cache_data(ttl=30)
def get_dashboard_data():

    # ----------------------------------------
    # Initialize Modules
    # ----------------------------------------

    digital_twin = get_digital_twin()

    decision_engine = get_decision_engine()

    # ----------------------------------------
    # Random Forest
    # ----------------------------------------

    health_result = digital_twin.predict_current_health()

    health = health_result["health"]

    health_confidence = int(round(health_result["confidence"]))

    # ----------------------------------------
    # LSTM
    # ----------------------------------------

    battery = digital_twin.predict_future_battery()

    # ----------------------------------------
    # Autoencoder
    # ----------------------------------------

    fault = digital_twin.detect_unknown_fault()
    
    # ----------------------------------------
    # Latest Sensor Values
    # ----------------------------------------

    sensor_values = digital_twin.get_latest_sensor_values()
    history = digital_twin.get_dashboard_history()
    shap_data = digital_twin.get_shap_importance()

    # ----------------------------------------
    # Decision Engine
    # ----------------------------------------

    decision_engine.load_ai_outputs(
        health_status=health,
        battery_prediction=battery,
        unknown_fault=fault["unknown_fault"]
    )

    overall = decision_engine.evaluate_overall_system(
        decision_engine.evaluate_system_health(),
        decision_engine.evaluate_unknown_fault(),
        decision_engine.evaluate_battery_soc(),
        decision_engine.evaluate_battery_soh(),
        decision_engine.evaluate_battery_rul()
    )

    recommendation = decision_engine.get_recommendation(
        overall["overall_status"]
    )

    alert = decision_engine.get_alert(
        overall["overall_status"]
    )
    
    now = datetime.now()

    # ----------------------------------------
    # Return Dashboard Data
    # ----------------------------------------

    return {

        "health": health,

        "health_confidence": health_confidence,

        "soc": round(battery["Battery_SOC"]),

        "soh": round(battery["Battery_SOH"]),

        "rul": round(battery["Battery_RUL"]),

        "unknown_fault": fault["unknown_fault"],

        "reconstruction_error": fault["reconstruction_error"],

        "digital_twin": "Synced",

        "recommendation": recommendation,

        "alert": alert,

        "overall_status": overall["overall_status"],
        
        "sensor_values": sensor_values,
        
        "history": history,
        
        "shap_data": shap_data,
        
        "last_updated_date": now.strftime("%d %b %Y"),

        "last_updated_time": now.strftime("%I:%M:%S %p")

    }