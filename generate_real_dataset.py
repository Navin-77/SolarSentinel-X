# =====================================================
# SolarSentinel-X
# Phase 13 : Real-Sensor Training Dataset Generator
# =====================================================
#
# IMPORTANT:
# This file is intended for the 10-feature REAL SENSOR
# training pipeline.
#
# The final research dataset should be collected from
# the ESP32 hardware and labelled according to known
# operating/fault conditions.
#
# This script currently defines the required schema and
# provides a safe structure for saving collected data.
# It does NOT generate fake training labels.
# =====================================================

from pathlib import Path
import pandas as pd


# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_FOLDER = BASE_DIR / "dataset" / "real_training"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "real_sensor_dataset.csv"


# -----------------------------------------------------
# REAL ESP32 SENSOR FEATURES
# -----------------------------------------------------

FEATURE_COLUMNS = [
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


# -----------------------------------------------------
# Dataset Columns
# -----------------------------------------------------

DATASET_COLUMNS = FEATURE_COLUMNS + [
    "System_Health"
]


# -----------------------------------------------------
# Create Empty Dataset Template
# -----------------------------------------------------

def create_dataset_template():
    """
    Create an empty CSV containing the exact 10-feature
    real-sensor schema plus the System_Health label.

    Real ESP32 measurements will be added later.
    """

    df = pd.DataFrame(columns=DATASET_COLUMNS)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 70)
    print("SOLARSENTINEL-X REAL SENSOR DATASET")
    print("=" * 70)

    print("\nDataset template created successfully!")
    print(f"Location : {OUTPUT_FILE}")

    print("\nReal Sensor Features:")
    for i, feature in enumerate(FEATURE_COLUMNS, start=1):
        print(f"{i:2d}. {feature}")

    print("\nTarget:")
    print("System_Health")

    print("\nNo synthetic values or artificial labels are generated.")


if __name__ == "__main__":
    create_dataset_template()
