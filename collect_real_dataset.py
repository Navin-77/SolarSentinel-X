import csv
import math
import time
from datetime import datetime
from pathlib import Path

import serial

from realtime.sensor_interface import parse_sensor_line


# ==========================================================
# CONFIGURATION
# ==========================================================

PORT = "COM11"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 2

TARGET_PER_CLASS = 100

OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / "dataset"
    / "real_training"
    / "new_real_sensor_dataset.csv"
)


# ==========================================================
# 10 REAL SENSOR FEATURES
# ==========================================================

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


# Timestamp is metadata.
# System_Health is the target label.
ALL_COLUMNS = [
    "Timestamp",
    *FEATURE_COLUMNS,
    "System_Health",
]


# ==========================================================
# SENSOR CONVERSION
# ==========================================================

def convert_sensor_data(parsed_data):

    return {
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


# ==========================================================
# VALIDATE SENSOR DATA
# ==========================================================

def validate_sensor_data(sensor_data):

    if not isinstance(sensor_data, dict):
        return False, "Invalid sensor data"

    # ------------------------------------------------------
    # Check that all 10 features exist
    # ------------------------------------------------------

    for feature in FEATURE_COLUMNS:

        if feature not in sensor_data:
            return False, f"Missing feature: {feature}"

        try:
            value = float(sensor_data[feature])
        except (TypeError, ValueError):
            return False, f"Non-numeric value: {feature}"

        # NaN
        if math.isnan(value):
            return False, f"NaN value: {feature}"

        # Infinity
        if math.isinf(value):
            return False, f"Infinite value: {feature}"

    # ------------------------------------------------------
    # DHT22 validation
    #
    # 0°C / 0% is treated as an invalid DHT22 reading.
    # We do NOT save it.
    # ------------------------------------------------------

    dht_temp = float(
        sensor_data["DHT22_Temperature"]
    )

    dht_humidity = float(
        sensor_data["DHT22_Humidity"]
    )

    if dht_temp == 0.0 and dht_humidity == 0.0:

        return (
            False,
            "Invalid DHT22 reading: temperature and humidity are both 0"
        )

    # ------------------------------------------------------
    # DHT22 physical sanity checks
    # ------------------------------------------------------

    if dht_humidity < 0 or dht_humidity > 100:

        return (
            False,
            f"Invalid DHT22 humidity: {dht_humidity}"
        )

    # ------------------------------------------------------
    # LDR ADC range
    # ESP32 ADC expected range
    # ------------------------------------------------------

    ldr = float(sensor_data["LDR"])

    if ldr < 0 or ldr > 4095:

        return (
            False,
            f"Invalid LDR value: {ldr}"
        )

    # ------------------------------------------------------
    # DS18B20
    #
    # Reject the common disconnected/error value -127°C.
    # ------------------------------------------------------

    ds18b20 = float(
        sensor_data["DS18B20_Temperature"]
    )

    if ds18b20 <= -126:

        return (
            False,
            f"Invalid DS18B20 reading: {ds18b20}"
        )

    return True, "Valid"


# ==========================================================
# GET EXISTING COUNTS
# ==========================================================

def get_existing_counts():

    counts = {
        "Critical": 0,
        "Warning": 0,
        "Healthy": 0,
    }

    if not OUTPUT_FILE.exists():
        return counts

    try:

        with open(
            OUTPUT_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                label = row.get(
                    "System_Health"
                )

                if label in counts:
                    counts[label] += 1

    except Exception as e:

        print(
            f"\nError reading existing CSV: {e}"
        )

        raise

    return counts


# ==========================================================
# CREATE CSV
# ==========================================================

def prepare_csv():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Create new file
    # ------------------------------------------------------

    if not OUTPUT_FILE.exists():

        with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=ALL_COLUMNS
            )

            writer.writeheader()

        print(
            "\nNew collection CSV created:"
        )

        print(OUTPUT_FILE)

        return

    # ------------------------------------------------------
    # Existing file
    # ------------------------------------------------------

    print(
        "\nExisting collection CSV found:"
    )

    print(OUTPUT_FILE)


# ==========================================================
# APPEND ONE VALID REAL READING
# ==========================================================

def append_row(sensor_data, label):

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    row = {
        "Timestamp": timestamp,
        "System_Health": label,
    }

    # Add all 10 sensor values
    for feature in FEATURE_COLUMNS:

        row[feature] = sensor_data[feature]

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=ALL_COLUMNS
        )

        writer.writerow(row)


# ==========================================================
# DISPLAY ALL 10 FEATURES
# ==========================================================

def display_sample(
    sample_number,
    label,
    sensor_data
):

    print("\n" + "-" * 100)

    print(
        f"ACCEPTED SAMPLE "
        f"{sample_number}/100 | "
        f"{label.upper()}"
    )

    print("-" * 100)

    print(
        f"LDR                  : "
        f"{sensor_data['LDR']:.2f}"
    )

    print(
        f"DHT22_Temperature    : "
        f"{sensor_data['DHT22_Temperature']:.2f} °C"
    )

    print(
        f"DHT22_Humidity       : "
        f"{sensor_data['DHT22_Humidity']:.2f} %"
    )

    print(
        f"DS18B20_Temperature  : "
        f"{sensor_data['DS18B20_Temperature']:.2f} °C"
    )

    print(
        f"Solar_Voltage        : "
        f"{sensor_data['Solar_Voltage']:.2f} V"
    )

    print(
        f"Solar_Current        : "
        f"{sensor_data['Solar_Current']:.2f} A"
    )

    print(
        f"Solar_Power          : "
        f"{sensor_data['Solar_Power']:.2f} W"
    )

    print(
        f"Battery_Voltage      : "
        f"{sensor_data['Battery_Voltage']:.2f} V"
    )

    print(
        f"Battery_Current      : "
        f"{sensor_data['Battery_Current']:.2f} A"
    )

    print(
        f"Battery_Power        : "
        f"{sensor_data['Battery_Power']:.2f} W"
    )

    print("-" * 100)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("SolarSentinel-X REAL SENSOR DATA COLLECTION")
    print("=" * 70)

    print(
        "\nONLY real ESP32 sensor measurements will be stored."
    )

    print(
        "No synthetic values will be generated."
    )

    print(
        "Existing real_sensor_dataset.csv will NOT be modified."
    )

    print(
        "\nTimestamp will be recorded for every accepted reading."
    )

    # ------------------------------------------------------
    # Prepare CSV
    # ------------------------------------------------------

    prepare_csv()

    # ------------------------------------------------------
    # Current counts
    # ------------------------------------------------------

    counts = get_existing_counts()

    print("\nCurrent collection status:")
    print(
        f"Critical : {counts['Critical']}/100"
    )
    print(
        f"Warning  : {counts['Warning']}/100"
    )
    print(
        f"Healthy  : {counts['Healthy']}/100"
    )
    print(
        f"Total    : "
        f"{sum(counts.values())}/300"
    )

    # ------------------------------------------------------
    # Select condition
    # ------------------------------------------------------

    print("\nSelect operating condition:")
    print("1. Critical")
    print("2. Warning")
    print("3. Healthy")

    choice = input(
        "\nEnter choice: "
    ).strip()

    label_map = {
        "1": "Critical",
        "2": "Warning",
        "3": "Healthy",
    }

    if choice not in label_map:

        print(
            "\nInvalid choice."
        )

        return

    label = label_map[choice]

    # ------------------------------------------------------
    # Existing samples
    # ------------------------------------------------------

    existing = counts[label]

    if existing >= TARGET_PER_CLASS:

        print(
            f"\n{label} already has "
            f"{existing} samples."
        )

        return

    remaining = (
        TARGET_PER_CLASS - existing
    )

    print("\n" + "=" * 70)

    print(
        f"COLLECTING {label.upper()} DATA"
    )

    print("=" * 70)

    print(
        f"Existing {label} samples : "
        f"{existing}"
    )

    print(
        f"Remaining required       : "
        f"{remaining}"
    )

    print(
        f"Final target             : "
        f"100"
    )

    print("\nIMPORTANT:")

    print(
        "1. Put the physical system into the intended condition."
    )

    print(
        "2. Close Arduino Serial Monitor."
    )

    print(
        "3. Keep ESP32 connected to COM11."
    )

    print(
        "4. Only valid readings will be saved."
    )

    print(
        "5. DHT22 0/0 readings will be rejected."
    )

    print(
        "6. All 10 sensor features will be displayed."
    )

    input(
        "\nPress ENTER when ready..."
    )

    # ------------------------------------------------------
    # Open serial
    # ------------------------------------------------------

    try:

        ser = serial.Serial(
            PORT,
            BAUD_RATE,
            timeout=SERIAL_TIMEOUT
        )

    except serial.SerialException as e:

        print(
            f"\nCould not open {PORT}."
        )

        print(
            f"Error: {e}"
        )

        return

    print(
        f"\nConnected to {PORT} "
        f"at {BAUD_RATE} baud."
    )

    print(
        "\nWaiting for valid ESP32 DATA lines..."
    )

    time.sleep(2)

    collected = 0
    invalid_lines = 0
    rejected_sensor_readings = 0

    # ------------------------------------------------------
    # COLLECTION LOOP
    # ------------------------------------------------------

    try:

        while collected < remaining:

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

            if not line.startswith(
                "DATA,"
            ):

                continue

            # --------------------------------------------------
            # Parse
            # --------------------------------------------------

            try:

                parsed = parse_sensor_line(
                    line
                )

                sensor_data = (
                    convert_sensor_data(
                        parsed
                    )
                )

            except Exception as e:

                invalid_lines += 1

                print(
                    "\n[REJECTED] "
                    f"Could not parse DATA line: {e}"
                )

                continue

            # --------------------------------------------------
            # Validate
            # --------------------------------------------------

            valid, reason = (
                validate_sensor_data(
                    sensor_data
                )
            )

            if not valid:

                rejected_sensor_readings += 1

                print(
                    "\n[REJECTED SENSOR READING]"
                )

                print(
                    f"Reason: {reason}"
                )

                continue

            # --------------------------------------------------
            # Save REAL reading
            # --------------------------------------------------

            append_row(
                sensor_data,
                label
            )

            collected += 1

            total_class_count = (
                existing + collected
            )

            display_sample(
                total_class_count,
                label,
                sensor_data
            )

    except KeyboardInterrupt:

        print(
            "\n\nCollection interrupted by user."
        )

        print(
            f"Accepted this session: "
            f"{collected}"
        )

        print(
            "All accepted readings are already saved."
        )

    finally:

        ser.close()

    # ------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------

    final_counts = (
        get_existing_counts()
    )

    print("\n" + "=" * 70)
    print("COLLECTION STATUS")
    print("=" * 70)

    print(
        f"Critical : "
        f"{final_counts['Critical']}/100"
    )

    print(
        f"Warning  : "
        f"{final_counts['Warning']}/100"
    )

    print(
        f"Healthy  : "
        f"{final_counts['Healthy']}/100"
    )

    print(
        f"Total    : "
        f"{sum(final_counts.values())}/300"
    )

    print(
        f"\nInvalid DATA lines ignored : "
        f"{invalid_lines}"
    )

    print(
        f"Invalid sensor readings    : "
        f"{rejected_sensor_readings}"
    )

    print(
        "\nSaved to:"
    )

    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()