import time
import random


def generate_sensor_data():
    """Generate simulated ESP32 sensor data."""

    sensor_data = {
        "ldr": random.uniform(600, 950),
        "dht22_temp": random.uniform(28, 36),
        "dht22_humidity": random.uniform(55, 80),
        "ds18b20_temp": random.uniform(30, 38),

        # Solar-side INA219
        "ina1_voltage": random.uniform(17, 21),
        "ina1_current": random.uniform(0.5, 3.0),
        "ina1_power": random.uniform(10, 60),

        # Battery-side INA219
        "ina2_voltage": random.uniform(11.5, 13.0),
        "ina2_current": random.uniform(0.2, 2.5),
        "ina2_power": random.uniform(5, 30),
    }

    return sensor_data


def convert_to_serial_format(data):
    """Convert sensor dictionary to the exact ESP32 DATA format."""

    return (
        f"DATA,"
        f"{data['ldr']:.0f},"
        f"{data['dht22_temp']:.2f},"
        f"{data['dht22_humidity']:.2f},"
        f"{data['ds18b20_temp']:.2f},"
        f"{data['ina1_voltage']:.3f},"
        f"{data['ina1_current']:.3f},"
        f"{data['ina1_power']:.3f},"
        f"{data['ina2_voltage']:.3f},"
        f"{data['ina2_current']:.3f},"
        f"{data['ina2_power']:.3f}"
    )


def main():

    print("=" * 60)
    print("SolarSentinel-X SENSOR SIMULATOR")
    print("=" * 60)
    print("Simulating ESP32 sensor output...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:

            data = generate_sensor_data()

            serial_line = convert_to_serial_format(data)

            print(serial_line)

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    main()