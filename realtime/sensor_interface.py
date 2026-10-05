def parse_sensor_line(line):
    """
    Convert ESP32 serial DATA line into a Python dictionary.

    Expected format:
    DATA,ldr,dht22_temp,dht22_humidity,ds18b20_temp,
    ina1_voltage,ina1_current,ina1_power,
    ina2_voltage,ina2_current,ina2_power
    """

    line = line.strip()

    if not line.startswith("DATA,"):
        return None

    values = line.split(",")

    if len(values) != 11:
        print("Invalid sensor data format")
        return None

    try:
        sensor_data = {
            "ldr": float(values[1]),
            "dht22_temp": float(values[2]),
            "dht22_humidity": float(values[3]),
            "ds18b20_temp": float(values[4]),

            "ina1_voltage": float(values[5]),
            "ina1_current": float(values[6]),
            "ina1_power": float(values[7]),

            "ina2_voltage": float(values[8]),
            "ina2_current": float(values[9]),
            "ina2_power": float(values[10]),
        }

        return sensor_data

    except ValueError:
        print("Invalid numeric sensor value")
        return None


if __name__ == "__main__":

    test_line = (
        "DATA,824,33.43,56.18,34.32,"
        "17.756,0.655,30.030,"
        "12.571,1.573,22.957"
    )

    result = parse_sensor_line(test_line)

    print("\nParsed Sensor Data")
    print("=" * 40)

    if result:
        for key, value in result.items():
            print(f"{key:20s}: {value}")

    print("=" * 40)