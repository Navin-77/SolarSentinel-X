import csv, math, time
from datetime import datetime
from pathlib import Path
import serial
from realtime.sensor_interface import parse_sensor_line

PORT = "COM11"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 2
TARGET = 100
LABEL = "Critical"

OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / "dataset"
    / "real_training"
    / "critical_nan_dataset.csv"
)

FEATURES = [
    "LDR", "DHT22_Temperature", "DHT22_Humidity",
    "DS18B20_Temperature", "Solar_Voltage", "Solar_Current",
    "Solar_Power", "Battery_Voltage", "Battery_Current",
    "Battery_Power"
]

COLUMNS = ["Timestamp", *FEATURES, "System_Health"]


def convert(parsed):
    return {
        "LDR": parsed["ldr"],
        "DHT22_Temperature": parsed["dht22_temp"],
        "DHT22_Humidity": parsed["dht22_humidity"],
        "DS18B20_Temperature": parsed["ds18b20_temp"],
        "Solar_Voltage": parsed["ina1_voltage"],
        "Solar_Current": parsed["ina1_current"],
        "Solar_Power": parsed["ina1_power"],
        "Battery_Voltage": parsed["ina2_voltage"],
        "Battery_Current": parsed["ina2_current"],
        "Battery_Power": parsed["ina2_power"],
    }


def validate(d):
    # All sensors except DS18B20 must be real, finite values.
    for feature in FEATURES:
        if feature == "DS18B20_Temperature":
            continue

        try:
            value = float(d[feature])
        except (TypeError, ValueError):
            return False, f"Invalid: {feature}"

        if not math.isfinite(value):
            return False, f"Invalid: {feature}"

    # DHT22 sanity checks
    dht_temp = float(d["DHT22_Temperature"])
    humidity = float(d["DHT22_Humidity"])

    if dht_temp == 0.0 and humidity == 0.0:
        return False, "DHT22 0/0"

    if not 0 <= humidity <= 100:
        return False, "Invalid DHT22 humidity"

    # ESP32 ADC range
    ldr = float(d["LDR"])
    if not 0 <= ldr <= 4095:
        return False, "Invalid LDR"

    # Critical collection intentionally allows DS18B20 to be unavailable.
    try:
        ds = float(d["DS18B20_Temperature"])
        if not math.isfinite(ds):
            raise ValueError
    except (TypeError, ValueError):
        d["DS18B20_Temperature"] = float("nan")

    return True, "Valid"


def prepare_csv():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def existing_count():
    if not OUTPUT_FILE.exists():
        return 0

    with open(OUTPUT_FILE, newline="", encoding="utf-8") as f:
        return sum(
            1 for row in csv.DictReader(f)
            if row.get("System_Health") == LABEL
        )


def append_row(d):
    row = {
        "Timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "System_Health": LABEL,
    }

    for feature in FEATURES:
        value = d[feature]

        if feature == "DS18B20_Temperature":
            try:
                value = float(value)
                if not math.isfinite(value):
                    value = "NaN"
            except (TypeError, ValueError):
                value = "NaN"

        row[feature] = value

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)


def display(d, number):
    print("\n" + "-" * 78)
    print(f"ACCEPTED CRITICAL SAMPLE {number}/{TARGET}")
    print("-" * 78)

    print(f"LDR                 : {float(d['LDR']):.2f}")
    print(f"DHT22 Temperature   : {float(d['DHT22_Temperature']):.2f} C")
    print(f"DHT22 Humidity      : {float(d['DHT22_Humidity']):.2f} %")

    try:
        ds = float(d["DS18B20_Temperature"])
        ds_text = f"{ds:.2f} C" if math.isfinite(ds) else "NaN"
    except (TypeError, ValueError):
        ds_text = "NaN"

    print(f"DS18B20 Temperature : {ds_text}")
    print(f"Solar Voltage       : {float(d['Solar_Voltage']):.6f} V")
    print(f"Solar Current       : {float(d['Solar_Current']):.6f}")
    print(f"Solar Power         : {float(d['Solar_Power']):.6f}")
    print(f"Battery Voltage     : {float(d['Battery_Voltage']):.6f} V")
    print(f"Battery Current     : {float(d['Battery_Current']):.6f}")
    print(f"Battery Power       : {float(d['Battery_Power']):.6f}")

    print("-" * 78)
    print("Other 9 values = REAL ESP32 readings")
    print("DS18B20 = allowed NaN")
    print("-" * 78)


def main():
    prepare_csv()

    existing = existing_count()

    print("=" * 70)
    print("SolarSentinel-X CRITICAL NaN DATA COLLECTION")
    print("=" * 70)
    print(f"Existing Critical samples : {existing}/100")
    print("Separate CSV              : critical_nan_dataset.csv")
    print("DS18B20 NaN               : allowed")
    print("Other 9 sensors           : must be valid real readings")
    print("=" * 70)

    if existing >= TARGET:
        print("Critical dataset already contains 100 samples.")
        return

    input(
        "\nPut the complete hardware into the intended CRITICAL "
        "experimental condition, close Arduino Serial Monitor, "
        "then press ENTER..."
    )

    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
    except serial.SerialException as e:
        print(f"\nCould not open {PORT}: {e}")
        return

    print(f"\nConnected to {PORT}. Collecting Critical data...")

    time.sleep(2)

    collected = 0

    try:
        while existing + collected < TARGET:
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line or not line.startswith("DATA,"):
                continue

            try:
                parsed = parse_sensor_line(line)

                if parsed is None:
                    raise ValueError("Parser returned None")

                sensor_data = convert(parsed)

            except Exception as e:
                print(f"[REJECTED DATA] {e}")
                continue

            valid, reason = validate(sensor_data)

            if not valid:
                print(f"[REJECTED SENSOR] {reason}")
                continue

            append_row(sensor_data)
            collected += 1
            display(sensor_data, existing + collected)

    except KeyboardInterrupt:
        print("\nCollection stopped.")
        print(f"Accepted this session: {collected}")

    finally:
        ser.close()

    print("\n" + "=" * 70)
    print("CRITICAL COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Critical samples : {existing_count()}/100")
    print(f"Saved to         : {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
