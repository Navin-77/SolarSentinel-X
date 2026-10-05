"""SolarSentinel-X V3 dataset-driven visual state helpers.

The visual state is derived from the existing real CSV through dataset_profile.py.
It is visualization only and is NOT an AI health diagnosis.
"""

from pathlib import Path


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


def sensor_snapshot(sensor_data):
    out = {}
    for name in FEATURES:
        try:
            out[name] = float(sensor_data.get(name, 0.0))
        except (TypeError, ValueError):
            out[name] = 0.0
    return out


def load_visual_profile():
    """Load the project's existing dataset-derived profile."""
    try:
        from dataset_profile import load_profile
        return load_profile()
    except Exception:
        return None


def visual_state(value, feature, profile):
    """Return LOW/NORMAL/HIGH using the existing dataset_profile logic."""
    if profile is None:
        return "NORMAL"

    try:
        from dataset_profile import classify_visual
        return str(classify_visual(value, feature, profile)).upper()
    except Exception:
        return "NORMAL"


def build_visual_states(data, profile):
    return {
        feature: visual_state(data[feature], feature, profile)
        for feature in FEATURES
    }


def state_color(state):
    """Color for visualization state only."""
    state = str(state).upper()
    if state == "LOW":
        return "#38BDF8"
    if state == "HIGH":
        return "#F59E0B"
    if state == "NORMAL":
        return "#22C55E"
    return "#94A3B8"


def state_symbol(state):
    state = str(state).upper()
    return {
        "LOW": "LOW",
        "NORMAL": "NORMAL",
        "HIGH": "HIGH",
    }.get(state, "UNAVAILABLE")
