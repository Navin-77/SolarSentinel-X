"""
SolarSentinel-X
Specific Diagnostic Recommendation Engine

Flow:
AI Predictions
    ↓
Sensor Pattern
    ↓
Temporal Trend
    ↓
Subsystem Mapping
    ↓
Deviation Analysis
    ↓
Evidence Strength
    ↓
Specific Diagnostic Recommendation
"""

from typing import Dict, Any, List


class DiagnosticRecommendationEngine:

    # ---------------------------------------------------------
    # SENSOR → SUBSYSTEM → COMPONENT MAPPING
    # ---------------------------------------------------------

    SUBSYSTEM_MAP = {
        "LDR": {
            "subsystem": "Solar Generation",
            "component": "Solar panel illumination sensing",
            "inspection": "Verify panel illumination, LDR placement and LDR sensing path."
        },

        "Solar_Voltage": {
            "subsystem": "Solar Generation",
            "component": "Solar voltage measurement",
            "inspection": "Verify solar-side voltage measurement and panel-side electrical connection."
        },

        "Solar_Current": {
            "subsystem": "Solar Generation",
            "component": "Solar current measurement",
            "inspection": "Verify solar-side current measurement, INA219 connection and panel-side current path."
        },

        "Solar_Power": {
            "subsystem": "Solar Generation",
            "component": "Solar power measurement",
            "inspection": "Verify solar generation and consistency between solar voltage and current measurements."
        },

        "Battery_Voltage": {
            "subsystem": "Battery",
            "component": "Battery voltage measurement",
            "inspection": "Verify battery voltage sensing and battery-side electrical connections."
        },

        "Battery_Current": {
            "subsystem": "Battery",
            "component": "Battery current measurement",
            "inspection": "Verify battery current sensing and INA219 battery-side connection."
        },

        "Battery_Power": {
            "subsystem": "Battery",
            "component": "Battery power behaviour",
            "inspection": "Check battery-side power behaviour and compare the trend with battery voltage/current."
        },

        "DHT22_Temperature": {
            "subsystem": "Thermal / Environmental",
            "component": "DHT22 temperature sensing",
            "inspection": "Verify DHT22 placement, wiring and environmental conditions."
        },

        "DHT22_Humidity": {
            "subsystem": "Thermal / Environmental",
            "component": "DHT22 humidity sensing",
            "inspection": "Verify humidity sensing conditions, DHT22 placement and sensor connection."
        },

        "DS18B20_Temperature": {
            "subsystem": "Thermal / Environmental",
            "component": "DS18B20 temperature sensing",
            "inspection": "Verify DS18B20 placement, wiring and 4.7kΩ pull-up connection."
        }
    }

    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------

    def __init__(self):
        self.last_result = None

    # ---------------------------------------------------------
    # NORMALIZE HEALTH STATE
    # ---------------------------------------------------------

    def _normalize_health(self, value):

        if value is None:
            return None

        value = str(value).upper().strip()

        if "CRITICAL" in value:
            return "CRITICAL"

        if "WARNING" in value or "WARN" in value:
            return "WARNING"

        if "HEALTHY" in value or "NORMAL" in value:
            return "HEALTHY"

        return value

    # ---------------------------------------------------------
    # EXTRACT HEALTH FROM MODEL RESULT
    # ---------------------------------------------------------

    def _extract_health(self, result):

        if not isinstance(result, dict):
            return None

        possible_keys = [
            "prediction",
            "predicted_health",
            "health",
            "status",
            "class"
        ]

        for key in possible_keys:

            if key in result:
                value = result.get(key)

                if value is not None:
                    return self._normalize_health(value)

        return None

    # ---------------------------------------------------------
    # EXTRACT SENSOR PATTERN STATUS
    # ---------------------------------------------------------

    def _get_sensor_pattern(self, sensor, sensor_pattern):

        if not isinstance(sensor_pattern, dict):
            return None

        abnormal_sensors = sensor_pattern.get("abnormal_sensors", [])

        if isinstance(abnormal_sensors, list):

            for item in abnormal_sensors:

                if not isinstance(item, dict):
                    continue

                if item.get("sensor") == sensor:
                    return item

        all_sensors = sensor_pattern.get("all_sensors", {})

        if isinstance(all_sensors, dict):

            item = all_sensors.get(sensor)

            if isinstance(item, dict):
                return item

        return None

    # ---------------------------------------------------------
    # EXTRACT TEMPORAL TREND
    # ---------------------------------------------------------

    def _get_trend(self, sensor, trend_summary):

        if not isinstance(trend_summary, dict):
            return "INSUFFICIENT_DATA"

        increasing = trend_summary.get("increasing", [])
        decreasing = trend_summary.get("decreasing", [])

        if sensor in increasing:
            return "INCREASING"

        if sensor in decreasing:
            return "DECREASING"

        return "STABLE"

    # ---------------------------------------------------------
    # EVIDENCE STRENGTH
    # ---------------------------------------------------------

    def _calculate_evidence(
        self,
        rf_health,
        lstm_health,
        ae_result,
        sensor_status,
        trend
    ):

        score = 0
        evidence = []

        # Random Forest
        if rf_health == "CRITICAL":
            score += 2
            evidence.append("Random Forest indicates CRITICAL health.")

        elif rf_health == "WARNING":
            score += 1
            evidence.append("Random Forest indicates WARNING health.")

        # LSTM
        if lstm_health == "CRITICAL":
            score += 2
            evidence.append("LSTM indicates CRITICAL temporal behaviour.")

        elif lstm_health == "WARNING":
            score += 1
            evidence.append("LSTM indicates WARNING temporal behaviour.")

        # Autoencoder
        ae_anomaly = False

        if isinstance(ae_result, dict):

            status = str(
                ae_result.get("status", "")
            ).upper()

            if "ANOMALY" in status:
                ae_anomaly = True

        if ae_anomaly:
            score += 2
            evidence.append(
                "Autoencoder detected an anomalous sensor pattern."
            )

        # Sensor pattern
        if sensor_status:

            status = str(
                sensor_status.get("status", "")
            ).upper()

            if status == "UNUSUAL":
                score += 2
                evidence.append(
                    "Sensor value is outside its learned operating pattern."
                )

            elif status == "DEVIATING":
                score += 1
                evidence.append(
                    "Sensor value shows measurable deviation from its learned pattern."
                )

        # Temporal behaviour
        if trend in ["INCREASING", "DECREASING"]:

            score += 1

            evidence.append(
                f"Temporal trend is {trend.lower()}."
            )

        # Evidence classification
        if score >= 7:
            strength = "STRONG"

        elif score >= 4:
            strength = "MODERATE"

        elif score >= 2:
            strength = "EARLY"

        else:
            strength = "LOW"

        return {
            "score": score,
            "strength": strength,
            "evidence": evidence
        }

    # ---------------------------------------------------------
    # SENSOR DIAGNOSTIC MESSAGE
    # ---------------------------------------------------------

    def _build_sensor_message(
        self,
        sensor,
        sensor_pattern,
        trend,
        evidence_strength
    ):

        if sensor_pattern is None:

            return {
                "sensor": sensor,
                "message": "No sensor-pattern evidence available.",
                "value": None,
                "status": "UNKNOWN",
                "deviation_percent": None,
                "z_score": None,
                "trend": trend,
                "evidence_strength": evidence_strength
            }

        value = sensor_pattern.get("value")
        status = sensor_pattern.get("status", "UNKNOWN")
        deviation = sensor_pattern.get("deviation_percent")
        z_score = sensor_pattern.get("z_score")

        if status == "UNUSUAL":

            message = (
                f"{sensor} is outside the learned operating pattern. "
                f"Check this sensing path before concluding that the "
                f"physical subsystem has failed."
            )

        elif status == "DEVIATING":

            message = (
                f"{sensor} is deviating from its learned operating pattern. "
                f"Continue monitoring and verify if the deviation persists."
            )

        else:

            message = (
                f"{sensor} is currently within the learned operating pattern."
            )

        return {
            "sensor": sensor,
            "message": message,
            "value": value,
            "status": status,
            "deviation_percent": deviation,
            "z_score": z_score,
            "trend": trend,
            "evidence_strength": evidence_strength
        }

    # ---------------------------------------------------------
    # SUBSYSTEM RECOMMENDATION
    # ---------------------------------------------------------

    def _build_subsystem_recommendation(
        self,
        subsystem,
        diagnostics
    ):

        sensors = diagnostics.get(subsystem, [])

        if not sensors:
            return "Continue monitoring."

        abnormal = [
            item for item in sensors
            if item.get("status") in [
                "UNUSUAL",
                "DEVIATING"
            ]
        ]

        if subsystem == "Solar Generation":

            if abnormal:

                return (
                    "Solar-generation indicators require targeted inspection. "
                    "Verify panel illumination, solar-side voltage/current "
                    "measurement and the consistency of solar power behaviour."
                )

            return (
                "Solar-generation indicators do not currently provide strong "
                "sensor-pattern evidence of abnormality."
            )

        if subsystem == "Battery":

            if abnormal:

                return (
                    "Battery-side behaviour requires targeted inspection. "
                    "Verify battery voltage/current sensing and compare "
                    "battery power behaviour with the recent temporal trend."
                )

            return (
                "Battery indicators do not currently provide strong "
                "sensor-pattern evidence of abnormality."
            )

        if subsystem == "Thermal / Environmental":

            if abnormal:

                return (
                    "Thermal/environmental sensing requires verification. "
                    "Check sensor placement, wiring and environmental "
                    "conditions before interpreting the deviation as a "
                    "physical subsystem fault."
                )

            return (
                "Thermal/environmental indicators are currently within "
                "their learned pattern."
            )

        return "Continue monitoring the affected subsystem."

    # ---------------------------------------------------------
    # MAIN ANALYSIS
    # ---------------------------------------------------------

    def analyze(
        self,
        rf_result=None,
        lstm_result=None,
        ae_result=None,
        sensor_pattern=None,
        trend_summary=None
    ):

        rf_health = self._extract_health(rf_result)
        lstm_health = self._extract_health(lstm_result)

        diagnostics = {}
        sensor_diagnostics = []

        abnormal_sensors = []

        if isinstance(sensor_pattern, dict):

            abnormal_sensors = sensor_pattern.get(
                "abnormal_sensors",
                []
            )

        # -----------------------------------------------------
        # Analyze each abnormal sensor
        # -----------------------------------------------------

        for item in abnormal_sensors:

            if not isinstance(item, dict):
                continue

            sensor = item.get("sensor")

            if sensor not in self.SUBSYSTEM_MAP:
                continue

            mapping = self.SUBSYSTEM_MAP[sensor]

            subsystem = mapping["subsystem"]

            trend = self._get_trend(
                sensor,
                trend_summary
            )

            evidence = self._calculate_evidence(
                rf_health=rf_health,
                lstm_health=lstm_health,
                ae_result=ae_result,
                sensor_status=item,
                trend=trend
            )

            diagnostic = self._build_sensor_message(
                sensor=sensor,
                sensor_pattern=item,
                trend=trend,
                evidence_strength=evidence["strength"]
            )

            diagnostic.update({

                "subsystem": subsystem,

                "component": mapping["component"],

                "inspection": mapping["inspection"],

                "evidence_score": evidence["score"],

                "evidence": evidence["evidence"]
            })

            sensor_diagnostics.append(diagnostic)

            if subsystem not in diagnostics:
                diagnostics[subsystem] = []

            diagnostics[subsystem].append(
                diagnostic
            )

        # -----------------------------------------------------
        # If no abnormal sensor was detected
        # -----------------------------------------------------

        if not sensor_diagnostics:

            result = {

                "diagnostic_status": "MONITOR",

                "evidence_strength": "LOW",

                "affected_subsystems": [],

                "sensor_diagnostics": [],

                "recommendation": (
                    "No specific sensor-level abnormality was identified "
                    "from the current CSV-derived operating pattern. "
                    "Continue monitoring the system."
                ),

                "inspection_points": []
            }

            self.last_result = result

            return result

        # -----------------------------------------------------
        # Determine strongest evidence
        # -----------------------------------------------------

        strength_order = {
            "LOW": 0,
            "EARLY": 1,
            "MODERATE": 2,
            "STRONG": 3
        }

        strongest = "LOW"

        for item in sensor_diagnostics:

            current = item.get(
                "evidence_strength",
                "LOW"
            )

            if strength_order.get(
                current,
                0
            ) > strength_order.get(
                strongest,
                0
            ):

                strongest = current

        # -----------------------------------------------------
        # Diagnostic status
        # -----------------------------------------------------

        if strongest == "STRONG":

            diagnostic_status = (
                "HIGH_PRIORITY_INSPECTION"
            )

        elif strongest == "MODERATE":

            diagnostic_status = (
                "TARGETED_INSPECTION"
            )

        elif strongest == "EARLY":

            diagnostic_status = (
                "EARLY_WARNING"
            )

        else:

            diagnostic_status = "MONITOR"

        # -----------------------------------------------------
        # Affected subsystems
        # -----------------------------------------------------

        affected_subsystems = list(
            diagnostics.keys()
        )

        # -----------------------------------------------------
        # Subsystem recommendations
        # -----------------------------------------------------

        subsystem_recommendations = {}

        for subsystem in affected_subsystems:

            subsystem_recommendations[subsystem] = (
                self._build_subsystem_recommendation(
                    subsystem,
                    diagnostics
                )
            )

        # -----------------------------------------------------
        # Inspection points
        # -----------------------------------------------------

        inspection_points = []

        for item in sensor_diagnostics:

            point = item.get("inspection")

            if point and point not in inspection_points:

                inspection_points.append(point)

        # -----------------------------------------------------
        # Overall recommendation
        # -----------------------------------------------------

        if strongest == "STRONG":

            recommendation = (
                "Multiple evidence sources indicate that targeted "
                "inspection should be performed immediately. "
                "Use the sensor-level diagnostics below to identify "
                "the affected subsystem and verify the sensing path."
            )

        elif strongest == "MODERATE":

            recommendation = (
                "The current evidence supports targeted inspection "
                "of the affected subsystem. Verify the abnormal "
                "sensor readings and their recent trend before "
                "concluding that a physical fault exists."
            )

        elif strongest == "EARLY":

            recommendation = (
                "Early deviation has been detected. Continue "
                "monitoring the affected sensors and verify whether "
                "the observed pattern persists."
            )

        else:

            recommendation = (
                "Evidence is currently weak. Continue monitoring "
                "the affected sensor behaviour."
            )

        # -----------------------------------------------------
        # Final result
        # -----------------------------------------------------

        result = {

            "diagnostic_status": diagnostic_status,

            "evidence_strength": strongest,

            "affected_subsystems": affected_subsystems,

            "sensor_diagnostics": sensor_diagnostics,

            "subsystem_recommendations":
                subsystem_recommendations,

            "recommendation": recommendation,

            "inspection_points": inspection_points
        }

        self.last_result = result

        return result