# =====================================================
# SolarSentinel-X
# Phase 13B : Real Sensor Data Logger
# =====================================================

from pathlib import Path
from datetime import datetime
import csv

from sensor_interface import parse_sensor_line


# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FOLDER = BASE_DIR / "dataset" / "real_training"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "real_sensor_dataset.csv"


# -----------------------------------------------------
# Dataset Columns
# -----------------------------------------------------

COLUMNS = [
    "Timestamp",
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
    "System_Health"
]


# -----------------------------------------------------
# Save One Sensor Reading
# -----------------------------------------------------

def save_sensor_data(sensor_data, system_health="Unknown"):

    file_exists = OUTPUT_FILE.exists()

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS
        )

        if not file_exists:
            writer.writeheader()

        row = {
            "Timestamp": datetime.now().isoformat(timespec="seconds"),

            "LDR": sensor_data["ldr"],
            "DHT22_Temperature": sensor_data["dht22_temp"],
            "DHT22_Humidity": sensor_data["dht22_humidity"],
            "DS18B20_Temperature": sensor_data["ds18b20_temp"],

            "Solar_Voltage": sensor_data["ina1_voltage"],
            "Solar_Current": sensor_data["ina1_current"],
            "Solar_Power": sensor_data["ina1_power"],

            "Battery_Voltage": sensor_data["ina2_voltage"],
            "Battery_Current": sensor_data["ina2_current"],
            "Battery_Power": sensor_data["ina2_power"],

            "System_Health": system_health
        }

        writer.writerow(row)


# -----------------------------------------------------
# Test With Simulator Output
# -----------------------------------------------------

def test_logger():

    print("=" * 70)
    print("SOLARSENTINEL-X REAL SENSOR DATA LOGGER")
    print("=" * 70)

    print("\nTesting logger with simulated ESP32 data...")
    print("No real hardware is required for this test.\n")

    test_lines = [
        "DATA,824,33.43,56.18,34.32,17.756,0.655,30.030,12.571,1.573,22.957",

        "DATA,692,34.35,71.78,31.00,20.908,2.259,20.959,12.834,2.420,10.750",

        "DATA,934,35.08,61.78,30.23,18.037,1.528,51.407,12.945,1.042,17.240"
    ]

    for line in test_lines:

        sensor_data = parse_sensor_line(line)

        if sensor_data is None:
            print("Invalid sensor data")
            continue

        # -------------------------------------------------
        # IMPORTANT:
        # These are test rows only.
        # They are NOT real training labels.
        # -------------------------------------------------

        save_sensor_data(
            sensor_data,
            system_health="Unknown"
        )

        print("Saved sensor reading:")
        print(
            f"Solar Power   : {sensor_data['ina1_power']:.3f} W"
        )
        print(
            f"Battery Power : {sensor_data['ina2_power']:.3f} W"
        )
        print()


# -----------------------------------------------------
# Main
# -----------------------------------------------------

if __name__ == "__main__":
    test_logger()

    print("=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

    print(f"\nDataset file:")
    print(OUTPUT_FILE)