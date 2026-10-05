from sensor_pattern_analyzer import SensorPatternAnalyzer, FEATURES


print("=" * 70)
print("SOLARSENTINEL-X PREDICTIVE MAINTENANCE PATTERN TEST")
print("=" * 70)


# ============================================================
# LOAD ANALYZER
# ============================================================

analyzer = SensorPatternAnalyzer()


# ============================================================
# NORMAL BASELINE
# ============================================================

normal_data = {
    "LDR": 816,
    "DHT22_Temperature": 32.2,
    "DHT22_Humidity": 66.8,
    "DS18B20_Temperature": 31.44,
    "Solar_Voltage": 1.004,
    "Solar_Current": 4.3,
    "Solar_Power": 4.0,
    "Battery_Voltage": 0.532,
    "Battery_Current": -0.4,
    "Battery_Power": 0.0,
}


# ============================================================
# ABNORMAL SENSOR TEST
# Simulate LDR sensor behaving abnormally
# ============================================================

abnormal_data = normal_data.copy()

abnormal_data["LDR"] = 50


# ============================================================
# ANALYZE
# ============================================================

result = analyzer.analyze(abnormal_data)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("Overall Pattern:")
print(result["overall_status"])

print()
print("-" * 70)

print("SENSOR ANALYSIS")
print("-" * 70)

for sensor in result["all_sensors"]:

    print(
        f"{sensor['sensor']:25s} | "
        f"Value: {str(sensor['value']):10s} | "
        f"Status: {sensor['status']:10s} | "
        f"Severity: {sensor['severity']:6s}"
    )


print("-" * 70)

print()
print("ABNORMAL SENSORS")
print("-" * 70)

if result["abnormal_sensors"]:

    for sensor in result["abnormal_sensors"]:

        print(
            f"Sensor       : {sensor['sensor']}"
        )

        print(
            f"Value        : {sensor['value']}"
        )

        print(
            f"Status       : {sensor['status']}"
        )

        print(
            f"Severity     : {sensor['severity']}"
        )

        print(
            f"Z-Score      : {sensor['z_score']}"
        )

        print(
            f"Deviation    : {sensor['deviation_percent']}%"
        )

        print(
            f"Message      : {sensor['message']}"
        )

        print("-" * 50)

else:

    print("No abnormal sensors detected.")


print()
print("=" * 70)
print("TEST COMPLETED")
print("=" * 70)