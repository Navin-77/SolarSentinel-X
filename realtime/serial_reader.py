import serial
import time
import re
import math

PORT = "COM15"
BAUD_RATE = 115200


def read_sensor_data():

    ser = None

    try:
        ser = serial.Serial(
            PORT,
            BAUD_RATE,
            timeout=2
        )

        time.sleep(2)

        print("=" * 70)
        print("SolarSentinel-X REAL SENSOR READER")
        print("=" * 70)
        print(f"Connected to {PORT} @ {BAUD_RATE} baud")
        print("Reading REAL ESP32 sensor output...")
        print("Arduino Serial Monitor must be CLOSED.")
        print("=" * 70)

        # Current reading storage
        sensor_data = {
            "ldr": None,
            "dht22_temp": None,
            "dht22_humidity": None,
            "ds18b20_temp": None,
            "ina1_voltage": None,
            "ina1_current": None,
            "ina1_power": None,
            "ina2_voltage": None,
            "ina2_current": None,
            "ina2_power": None,
        }

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

            # --------------------------------------------------
            # LDR
            # --------------------------------------------------

            if line.startswith("RAW:"):
                line = line[4:].strip()

            if line.startswith("LDR Light Value"):

                match = re.search(
                    r":\s*(-?\d+(?:\.\d+)?)",
                    line
                )

                if match:
                    sensor_data["ldr"] = float(
                        match.group(1)
                    )

            # --------------------------------------------------
            # DHT22
            # --------------------------------------------------

            elif line.startswith("DHT22"):

                if "READ ERROR" in line:

                    sensor_data["dht22_temp"] = None
                    sensor_data["dht22_humidity"] = None

                else:

                    # Supports formats such as:
                    # DHT22 : 31.4 C, 76.0 %
                    # DHT22 : Temperature: 31.4 C Humidity: 76.0 %

                    temp_match = re.search(
                        r"(?:Temperature\s*:?\s*)?"
                        r"(-?\d+(?:\.\d+)?)\s*"
                        r"(?:C|°C)",
                        line,
                        re.IGNORECASE
                    )

                    humidity_match = re.search(
                        r"(?:Humidity\s*:?\s*)?"
                        r"(\d+(?:\.\d+)?)\s*%",
                        line,
                        re.IGNORECASE
                    )

                    if temp_match:
                        sensor_data["dht22_temp"] = float(
                            temp_match.group(1)
                        )

                    if humidity_match:
                        sensor_data["dht22_humidity"] = float(
                            humidity_match.group(1)
                        )

            # --------------------------------------------------
            # DS18B20
            # --------------------------------------------------

            elif line.startswith("DS18B20"):

                if "DISCONNECTED" in line:

                    sensor_data["ds18b20_temp"] = None

                else:

                    match = re.search(
                        r":\s*(-?\d+(?:\.\d+)?)\s*"
                        r"(?:C|°C)",
                        line,
                        re.IGNORECASE
                    )

                    if match:
                        sensor_data["ds18b20_temp"] = float(
                            match.group(1)
                        )

            # --------------------------------------------------
            # INA219 #1
            # --------------------------------------------------

            elif line.startswith("Bus Voltage"):

                match = re.search(
                    r":\s*(-?\d+(?:\.\d+)?)\s*V",
                    line,
                    re.IGNORECASE
                )

                if match:

                    voltage = float(
                        match.group(1)
                    )

                    # If we are inside the first INA block
                    if sensor_data["ina1_voltage"] is None:
                        sensor_data["ina1_voltage"] = voltage
                    else:
                        sensor_data["ina2_voltage"] = voltage

            # --------------------------------------------------
            # Current
            # --------------------------------------------------

            elif line.startswith("Current"):

                match = re.search(
                    r":\s*(-?\d+(?:\.\d+)?)\s*mA",
                    line,
                    re.IGNORECASE
                )

                if match:

                    current_ma = float(
                        match.group(1)
                    )

                    current_a = (
                        current_ma / 1000.0
                    )

                    if sensor_data["ina1_current"] is None:
                        sensor_data["ina1_current"] = current_a
                    else:
                        sensor_data["ina2_current"] = current_a

            # --------------------------------------------------
            # Power
            # --------------------------------------------------

            elif line.startswith("Power"):

                match = re.search(
                    r":\s*(-?\d+(?:\.\d+)?)\s*mW",
                    line,
                    re.IGNORECASE
                )

                if match:

                    power_mw = float(
                        match.group(1)
                    )

                    power_w = (
                        power_mw / 1000.0
                    )

                    if sensor_data["ina1_power"] is None:
                        sensor_data["ina1_power"] = power_w
                    else:
                        sensor_data["ina2_power"] = power_w

            # --------------------------------------------------
            # Detect end of reading
            # --------------------------------------------------

            elif "Next reading" in line:

                print("\n" + "=" * 70)
                print("CURRENT REAL SENSOR READING")
                print("=" * 70)

                print(
                    f"LDR                  : "
                    f"{sensor_data['ldr']}"
                )

                print(
                    f"DHT22 Temperature    : "
                    f"{sensor_data['dht22_temp']}"
                )

                print(
                    f"DHT22 Humidity       : "
                    f"{sensor_data['dht22_humidity']}"
                )

                print(
                    f"DS18B20 Temperature  : "
                    f"{sensor_data['ds18b20_temp']}"
                )

                print(
                    f"Solar Voltage        : "
                    f"{sensor_data['ina1_voltage']} V"
                )

                print(
                    f"Solar Current        : "
                    f"{sensor_data['ina1_current']} A"
                )

                print(
                    f"Solar Power          : "
                    f"{sensor_data['ina1_power']} W"
                )

                print(
                    f"Battery Voltage      : "
                    f"{sensor_data['ina2_voltage']} V"
                )

                print(
                    f"Battery Current      : "
                    f"{sensor_data['ina2_current']} A"
                )

                print(
                    f"Battery Power        : "
                    f"{sensor_data['ina2_power']} W"
                )

                print("=" * 70)

                # --------------------------------------------------
                # Sensor health status
                # --------------------------------------------------

                errors = []

                if sensor_data["dht22_temp"] is None:
                    errors.append("DHT22")

                if sensor_data["dht22_humidity"] is None:
                    errors.append("DHT22")

                if sensor_data["ds18b20_temp"] is None:
                    errors.append("DS18B20")

                if errors:

                    print(
                        "\nSTATUS: INVALID FOR DATASET COLLECTION"
                    )

                    print(
                        "Sensor errors: "
                        + ", ".join(
                            sorted(set(errors))
                        )
                    )

                    print(
                        "Fix the sensor connection/read error first."
                    )

                else:

                    # Basic validity
                    values = list(
                        sensor_data.values()
                    )

                    if all(
                        v is not None
                        and math.isfinite(float(v))
                        for v in values
                    ):

                        print(
                            "\nSTATUS: ALL 10 SENSOR VALUES VALID"
                        )

                        print(
                            "This reading can be evaluated "
                            "against your real dataset."
                        )

                # Reset for next reading
                sensor_data = {
                    "ldr": None,
                    "dht22_temp": None,
                    "dht22_humidity": None,
                    "ds18b20_temp": None,
                    "ina1_voltage": None,
                    "ina1_current": None,
                    "ina1_power": None,
                    "ina2_voltage": None,
                    "ina2_current": None,
                    "ina2_power": None,
                }

    except serial.SerialException as e:

        print(
            f"\nSerial connection error: {e}"
        )

        print(
            f"Check that ESP32 is connected to {PORT}."
        )

        print(
            "Also make sure Arduino Serial Monitor is CLOSED."
        )

    except KeyboardInterrupt:

        print("\n\nStopped by user.")

    finally:

        if ser is not None and ser.is_open:
            ser.close()


if __name__ == "__main__":
    read_sensor_data()