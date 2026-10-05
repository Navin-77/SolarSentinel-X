import os
import csv
import time
import serial
from datetime import datetime

from sensor_interface import parse_sensor_line


# ============================================================
# CONFIGURATION
# ============================================================

PORT = "COM11"
BAUD_RATE = 115200

DATASET_DIR = os.path.join("dataset", "real_training")
DATASET_FILE = os.path.join(DATASET_DIR, "real_sensor_dataset.csv")

SAMPLE_INTERVAL = 2          # seconds between saved samples
SAMPLES_PER_SESSION = 100    # samples collected in one session


# ============================================================
# DATASET COLUMNS
# ============================================================

CSV_COLUMNS = [
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


# ============================================================
# CREATE DATASET FILE
# ============================================================

def initialize_dataset():

    os.makedirs(DATASET_DIR, exist_ok=True)

    if not os.path.exists(DATASET_FILE):

        with open(
            DATASET_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS
            )

            writer.writeheader()

        print("Dataset file created.")


# ============================================================
# SAVE SENSOR SAMPLE
# ============================================================

def save_sample(sensor_data, health_label):

    row = {
        "Timestamp": datetime.now().isoformat(),

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

        "System_Health": health_label
    }

    with open(
        DATASET_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        writer.writerow(row)


# ============================================================
# DISPLAY SENSOR DATA
# ============================================================

def display_sensor_data(sensor_data, sample_number):

    print()
    print("-" * 65)
    print(f"Sample {sample_number}")

    print(f"LDR                 : {sensor_data['ldr']}")
    print(f"DHT22 Temperature    : {sensor_data['dht22_temp']:.2f} °C")
    print(f"DHT22 Humidity       : {sensor_data['dht22_humidity']:.2f} %")
    print(f"DS18B20 Temperature  : {sensor_data['ds18b20_temp']:.2f} °C")

    print(f"Solar Voltage        : {sensor_data['ina1_voltage']:.3f} V")
    print(f"Solar Current        : {sensor_data['ina1_current']:.3f} mA")
    print(f"Solar Power          : {sensor_data['ina1_power']:.3f} mW")

    print(f"Battery Voltage      : {sensor_data['ina2_voltage']:.3f} V")
    print(f"Battery Current      : {sensor_data['ina2_current']:.3f} mA")
    print(f"Battery Power        : {sensor_data['ina2_power']:.3f} mW")


# ============================================================
# COLLECT ONE SESSION
# ============================================================

def collect_session(health_label):

    print()
    print("=" * 65)
    print(f"COLLECTING: {health_label.upper()}")
    print("=" * 65)

    print(f"Target samples : {SAMPLES_PER_SESSION}")
    print(f"Interval       : {SAMPLE_INTERVAL} seconds")

    print()
    print("Connect the ESP32 and make sure Serial Monitor is CLOSED.")
    input("Press ENTER when ready to start collection...")

    try:

        ser = serial.Serial(
            PORT,
            BAUD_RATE,
            timeout=2
        )

    except Exception as error:

        print()
        print("ERROR: Could not open ESP32 serial port.")
        print(error)
        return 0

    print()
    print("ESP32 connected successfully.")
    print("Starting data collection...")
    print("Press CTRL+C to stop safely.")
    print()

    collected = 0

    try:

        while collected < SAMPLES_PER_SESSION:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            if not line.startswith("DATA,"):
                continue

            try:

                sensor_data = parse_sensor_line(line)

                if sensor_data is None:
                    continue

                collected += 1

                save_sample(
                    sensor_data,
                    health_label
                )

                display_sensor_data(
                    sensor_data,
                    collected
                )

                print(f"Label               : {health_label}")

                time.sleep(SAMPLE_INTERVAL)

            except Exception as error:

                print()
                print("Sensor parsing error:")
                print(error)

    except KeyboardInterrupt:

        print()
        print()
        print("Collection stopped by user.")

    finally:

        ser.close()

    print()
    print("=" * 65)
    print(f"{health_label.upper()} SESSION COMPLETED")
    print(f"Samples collected : {collected}")
    print("=" * 65)

    return collected


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 65)
    print("SOLARSENTINEL-X REAL SENSOR DATA COLLECTION")
    print("=" * 65)

    print()
    print("IMPORTANT:")
    print("This program records REAL ESP32 sensor measurements.")
    print("Do NOT use simulator data for final model training.")
    print()

    initialize_dataset()

    print("Select the experimental condition:")
    print()
    print("1. Healthy")
    print("2. Warning")
    print("3. Critical")
    print("4. Exit")

    choice = input(
        "\nEnter your choice (1-4): "
    ).strip()

    label_map = {
        "1": "Healthy",
        "2": "Warning",
        "3": "Critical"
    }

    if choice == "4":
        print("Exiting.")
        return

    if choice not in label_map:

        print("Invalid choice.")
        return

    health_label = label_map[choice]

    print()
    print("=" * 65)
    print(f"SELECTED CONDITION: {health_label.upper()}")
    print("=" * 65)

    print()
    print("IMPORTANT:")
    print()
    print("Only select this label when the physical system")
    print("is actually operating under the corresponding")
    print("experiment condition.")
    print()

    collect_session(health_label)

    print()
    print("Data saved to:")
    print(DATASET_FILE)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()