"""
SolarSentinel-X
Predictive Maintenance Engine

Converts sensor-pattern analysis into:
    - Severity
    - Affected sensor
    - Maintenance recommendation
    - Suggested inspection

Important:
This module provides condition-based recommendations.
It does NOT claim that a physical component has definitely failed.
"""

from pathlib import Path
import sys


# ============================================================
# ALLOW IMPORT FROM predictive_maintenance FOLDER
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


from sensor_pattern_analyzer import SensorPatternAnalyzer


# ============================================================
# SENSOR → COMPONENT MAPPING
# ============================================================

SENSOR_COMPONENT_MAP = {

    "LDR": {
        "component": "LDR / Light Sensing Circuit",
        "inspection": [
            "Check LDR sensor connection.",
            "Check sensor wiring and GPIO connection.",
            "Verify that the LDR is receiving expected light.",
            "Inspect the sensor for physical blockage or damage.",
        ],
    },

    "DHT22_Temperature": {
        "component": "DHT22 Temperature Sensing",
        "inspection": [
            "Check DHT22 sensor connection.",
            "Inspect the sensor wiring.",
            "Verify the sensor power supply.",
            "Check whether the temperature reading is physically reasonable.",
        ],
    },

    "DHT22_Humidity": {
        "component": "DHT22 Humidity Sensing",
        "inspection": [
            "Check DHT22 sensor connection.",
            "Inspect the sensor wiring.",
            "Verify the sensor power supply.",
            "Check for abnormal environmental conditions.",
        ],
    },

    "DS18B20_Temperature": {
        "component": "DS18B20 Temperature Sensing",
        "inspection": [
            "Check DS18B20 sensor connection.",
            "Inspect the DATA wire.",
            "Verify the pull-up resistor connection.",
            "Check the sensor for intermittent readings.",
        ],
    },

    "Solar_Voltage": {
        "component": "Solar-side INA219 / Voltage Measurement",
        "inspection": [
            "Check the solar-side INA219 connection.",
            "Verify SDA and SCL wiring.",
            "Check the solar measurement path.",
            "Verify that the voltage reading matches the physical system condition.",
        ],
    },

    "Solar_Current": {
        "component": "Solar-side INA219 / Current Measurement",
        "inspection": [
            "Check the solar-side INA219 connection.",
            "Inspect the current measurement path.",
            "Verify INA219 wiring.",
            "Check for unexpected changes in solar current.",
        ],
    },

    "Solar_Power": {
        "component": "Solar Power Measurement",
        "inspection": [
            "Check solar voltage and current measurements.",
            "Inspect the solar-side INA219.",
            "Verify sensor wiring.",
            "Check whether the solar source is producing expected power.",
        ],
    },

    "Battery_Voltage": {
        "component": "Battery-side INA219 / Battery Voltage Measurement",
        "inspection": [
            "Check the battery-side INA219 connection.",
            "Verify SDA and SCL wiring.",
            "Check the battery connection.",
            "Verify that the measured voltage matches the physical battery condition.",
        ],
    },

    "Battery_Current": {
        "component": "Battery-side INA219 / Current Measurement",
        "inspection": [
            "Check the battery-side INA219 connection.",
            "Inspect the current measurement path.",
            "Verify INA219 wiring.",
            "Check for unexpected charging or discharging behaviour.",
        ],
    },

    "Battery_Power": {
        "component": "Battery Power Measurement",
        "inspection": [
            "Check battery voltage and current measurements.",
            "Inspect the battery-side INA219.",
            "Verify battery-side wiring.",
            "Check the battery charging/discharging path.",
        ],
    },
}


# ============================================================
# MAINTENANCE ENGINE
# ============================================================

class MaintenanceEngine:

    def __init__(self, analyzer=None):

        if analyzer is None:
            analyzer = SensorPatternAnalyzer()

        self.analyzer = analyzer


    # ========================================================
    # GENERATE RECOMMENDATION
    # ========================================================

    def generate(self, sensor_data):

        analysis = self.analyzer.analyze(sensor_data)

        abnormal_sensors = analysis["abnormal_sensors"]


        # ----------------------------------------------------
        # NO ABNORMALITY
        # ----------------------------------------------------

        if not abnormal_sensors:

            return {
                "status": "NORMAL",
                "severity": "LOW",
                "affected_sensor": "None",
                "affected_component": "None",
                "pattern": "Normal operating pattern",
                "maintenance_recommendation": (
                    "No immediate maintenance action is required. "
                    "Continue monitoring the system."
                ),
                "suggested_inspection": [
                    "Continue normal sensor monitoring."
                ],
                "evidence": [],
            }


        # ----------------------------------------------------
        # SELECT MOST SIGNIFICANT SENSOR
        # ----------------------------------------------------

        primary = abnormal_sensors[0]

        sensor_name = primary["sensor"]

        severity = primary["severity"]

        status = primary["status"]


        # ----------------------------------------------------
        # COMPONENT INFORMATION
        # ----------------------------------------------------

        component_info = SENSOR_COMPONENT_MAP.get(
            sensor_name,
            {
                "component": "Associated sensor/component",
                "inspection": [
                    "Check sensor connection.",
                    "Inspect wiring.",
                    "Verify sensor output.",
                ],
            }
        )


        component = component_info["component"]

        inspection = component_info["inspection"]


        # ----------------------------------------------------
        # MAINTENANCE RECOMMENDATION
        # ----------------------------------------------------

        if status == "MISSING":

            recommendation = (
                f"No valid output is being received from "
                f"{sensor_name}. Check the sensor connection, "
                f"communication path and wiring."
            )

            overall_status = "MAINTENANCE_ALERT"


        elif severity == "HIGH":

            recommendation = (
                f"{sensor_name} shows a highly unusual pattern "
                f"relative to the learned training data. "
                f"Inspect the associated sensing path and "
                f"verify the physical condition of the component."
            )

            overall_status = "MAINTENANCE_ALERT"


        else:

            recommendation = (
                f"{sensor_name} is showing a deviation from "
                f"its learned operating pattern. Continue monitoring "
                f"and inspect the associated sensing path if "
                f"the deviation persists."
            )

            overall_status = "MONITOR"


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {

            "status": overall_status,

            "severity": severity,

            "affected_sensor": sensor_name,

            "affected_component": component,

            "pattern": status,

            "maintenance_recommendation": recommendation,

            "suggested_inspection": inspection,

            "evidence": [
                {
                    "sensor": primary["sensor"],
                    "value": primary["value"],
                    "status": primary["status"],
                    "severity": primary["severity"],
                    "z_score": primary["z_score"],
                    "deviation_percent": primary[
                        "deviation_percent"
                    ],
                    "message": primary["message"],
                }
            ],

            "additional_abnormal_sensors": [
                item["sensor"]
                for item in abnormal_sensors[1:]
            ],
        }


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 80)
    print("SOLARSENTINEL-X PREDICTIVE MAINTENANCE ENGINE")
    print("=" * 80)


    analyzer = SensorPatternAnalyzer()

    engine = MaintenanceEngine(analyzer)


    # --------------------------------------------------------
    # Simulated abnormal condition
    # --------------------------------------------------------

    test_data = {

        "LDR": 50,

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


    result = engine.generate(test_data)


    # ========================================================
    # DISPLAY
    # ========================================================

    print()

    print("STATUS")
    print(result["status"])

    print()

    print("SEVERITY")
    print(result["severity"])

    print()

    print("AFFECTED SENSOR")
    print(result["affected_sensor"])

    print()

    print("AFFECTED COMPONENT")
    print(result["affected_component"])

    print()

    print("PATTERN")
    print(result["pattern"])

    print()

    print("MAINTENANCE RECOMMENDATION")
    print(result["maintenance_recommendation"])

    print()

    print("SUGGESTED INSPECTION")

    for index, item in enumerate(
        result["suggested_inspection"],
        start=1
    ):

        print(f"{index}. {item}")


    print()

    print("EVIDENCE")

    for evidence in result["evidence"]:

        print(
            f"Sensor: {evidence['sensor']}"
        )

        print(
            f"Value: {evidence['value']}"
        )

        print(
            f"Z-Score: {evidence['z_score']}"
        )

        print(
            f"Deviation: "
            f"{evidence['deviation_percent']}%"
        )

    print()

    print("=" * 80)
    print("PREDICTIVE MAINTENANCE TEST COMPLETED")
    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()