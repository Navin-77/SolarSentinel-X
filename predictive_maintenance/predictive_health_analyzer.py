# ============================================================
# SolarSentinel-X
# Predictive Maintenance — Predictive Health Analyzer
# ============================================================

class PredictiveHealthAnalyzer:

    # --------------------------------------------------------
    # HEALTH PRIORITY
    # --------------------------------------------------------

    HEALTH_PRIORITY = {
        "CRITICAL": 3,
        "WARNING": 2,
        "NORMAL": 1,
        "UNKNOWN": 0
    }

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def __init__(self):

        self.last_result = None

    # --------------------------------------------------------
    # NORMALIZE HEALTH STATUS
    # --------------------------------------------------------

    def normalize_health(self, value):

        if value is None:
            return "UNKNOWN"

        value = str(value).upper().strip()

        if value in ["CRITICAL", "CRIT"]:
            return "CRITICAL"

        if value in ["WARNING", "WARN"]:
            return "WARNING"

        if value in ["HEALTHY", "NORMAL"]:
            return "NORMAL"

        return "UNKNOWN"

    # --------------------------------------------------------
    # EXTRACT RANDOM FOREST HEALTH
    # --------------------------------------------------------

    def get_rf_health(self, rf_result):

        if not rf_result:
            return "UNKNOWN"

        # Common possible field names
        for key in [
            "predicted_health",
            "prediction",
            "health",
            "status",
            "class"
        ]:

            if key in rf_result:
                return self.normalize_health(
                    rf_result[key]
                )

        return "UNKNOWN"

    # --------------------------------------------------------
    # EXTRACT LSTM HEALTH
    # --------------------------------------------------------

    def get_lstm_health(self, lstm_result):

        if not lstm_result:
            return "UNKNOWN"

        for key in [
            "predicted_health",
            "prediction",
            "health",
            "status",
            "class"
        ]:

            if key in lstm_result:
                return self.normalize_health(
                    lstm_result[key]
                )

        return "UNKNOWN"

    # --------------------------------------------------------
    # AUTOENCODER ANOMALY
    # --------------------------------------------------------

    def get_autoencoder_status(self, ae_result):

        if not ae_result:
            return "UNKNOWN"

        anomaly = ae_result.get("anomaly")

        if anomaly is True:
            return "ANOMALY"

        if anomaly is False:
            return "NORMAL"

        status = ae_result.get("status")

        if status:
            status = str(status).upper()

            if "ANOM" in status:
                return "ANOMALY"

            if "NORMAL" in status:
                return "NORMAL"

        return "UNKNOWN"

    # --------------------------------------------------------
    # SENSOR PATTERN STATUS
    # --------------------------------------------------------

    def get_sensor_pattern_status(self, pattern_result):

        if not pattern_result:
            return "UNKNOWN"

        status = pattern_result.get(
            "overall_status"
        )

        if status is None:
            return "UNKNOWN"

        status = str(status).upper()

        if "CRITICAL" in status:
            return "CRITICAL"

        if "ABNORMAL" in status:
            return "WARNING"

        if "UNUSUAL" in status:
            return "WARNING"

        if "DEVIAT" in status:
            return "WARNING"

        if "NORMAL" in status:
            return "NORMAL"

        return "UNKNOWN"

    # --------------------------------------------------------
    # TEMPORAL TREND EVIDENCE
    # --------------------------------------------------------

    def analyze_trend_evidence(self, trend_summary):

        if not trend_summary:
            return {
                "status": "UNKNOWN",
                "evidence": []
            }

        increasing = trend_summary.get(
            "increasing_sensors",
            []
        )

        decreasing = trend_summary.get(
            "decreasing_sensors",
            []
        )

        evidence = []

        for sensor in increasing:
            evidence.append(
                f"{sensor} is increasing"
            )

        for sensor in decreasing:
            evidence.append(
                f"{sensor} is decreasing"
            )

        if evidence:
            return {
                "status": "TREND_CHANGE",
                "evidence": evidence
            }

        return {
            "status": "STABLE",
            "evidence": []
        }

    # --------------------------------------------------------
    # PREDICTIVE HEALTH
    # --------------------------------------------------------

    def analyze(
        self,
        rf_result=None,
        lstm_result=None,
        ae_result=None,
        sensor_pattern=None,
        trend_summary=None
    ):

        rf_health = self.get_rf_health(
            rf_result
        )

        lstm_health = self.get_lstm_health(
            lstm_result
        )

        ae_status = self.get_autoencoder_status(
            ae_result
        )

        pattern_status = self.get_sensor_pattern_status(
            sensor_pattern
        )

        trend_result = self.analyze_trend_evidence(
            trend_summary
        )

        # ----------------------------------------------------
        # COLLECT HEALTH EVIDENCE
        # ----------------------------------------------------

        health_states = [
            rf_health,
            lstm_health,
            pattern_status
        ]

        # Remove unavailable results
        health_states = [
            state
            for state in health_states
            if state != "UNKNOWN"
        ]

        # ----------------------------------------------------
        # DETERMINE PREDICTIVE HEALTH
        # ----------------------------------------------------

        if "CRITICAL" in health_states:

            predictive_health = "CRITICAL"

        elif "WARNING" in health_states:

            predictive_health = "WARNING"

        elif ae_status == "ANOMALY":

            predictive_health = "WARNING"

        else:

            predictive_health = "NORMAL"

        # ----------------------------------------------------
        # CONFIDENCE / AGREEMENT
        # ----------------------------------------------------

        critical_count = health_states.count(
            "CRITICAL"
        )

        warning_count = health_states.count(
            "WARNING"
        )

        normal_count = health_states.count(
            "NORMAL"
        )

        if critical_count >= 2:

            evidence_level = "STRONG"

        elif warning_count >= 2:

            evidence_level = "MODERATE"

        elif ae_status == "ANOMALY":

            evidence_level = "MODERATE"

        elif trend_result["status"] == "TREND_CHANGE":

            evidence_level = "EARLY"

        else:

            evidence_level = "LOW"

        # ----------------------------------------------------
        # PREDICTIVE INDICATORS
        # ----------------------------------------------------

        indicators = []

        if rf_health != "UNKNOWN":
            indicators.append(
                f"Random Forest: {rf_health}"
            )

        if lstm_health != "UNKNOWN":
            indicators.append(
                f"LSTM: {lstm_health}"
            )

        if ae_status != "UNKNOWN":
            indicators.append(
                f"Autoencoder: {ae_status}"
            )

        if pattern_status != "UNKNOWN":
            indicators.append(
                f"Sensor pattern: {pattern_status}"
            )

        if trend_result["status"] == "TREND_CHANGE":

            indicators.extend(
                trend_result["evidence"]
            )

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        if predictive_health == "CRITICAL":

            recommendation = (
                "Immediate inspection and "
                "maintenance are recommended."
            )

        elif predictive_health == "WARNING":

            recommendation = (
                "Inspect the system and monitor "
                "the abnormal or changing conditions."
            )

        else:

            recommendation = (
                "System condition is currently normal. "
                "Continue monitoring."
            )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = {

            "predictive_health":
                predictive_health,

            "evidence_level":
                evidence_level,

            "random_forest":
                rf_health,

            "lstm":
                lstm_health,

            "autoencoder":
                ae_status,

            "sensor_pattern":
                pattern_status,

            "temporal_trend":
                trend_result,

            "indicators":
                indicators,

            "recommendation":
                recommendation
        }

        self.last_result = result

        return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    analyzer = PredictiveHealthAnalyzer()

    # --------------------------------------------------------
    # Example model outputs
    # --------------------------------------------------------

    rf_result = {
        "predicted_health": "Warning"
    }

    lstm_result = {
        "predicted_health": "Warning"
    }

    ae_result = {
        "anomaly": True
    }

    sensor_pattern = {
        "overall_status": "ABNORMAL"
    }

    trend_summary = {
        "increasing_sensors": [
            "DHT22_Temperature"
        ],
        "decreasing_sensors": [
            "Solar_Power"
        ],
        "stable_sensors": [
            "LDR"
        ],
        "insufficient_data": [],
        "total_sensors": 10
    }

    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------

    result = analyzer.analyze(
        rf_result=rf_result,
        lstm_result=lstm_result,
        ae_result=ae_result,
        sensor_pattern=sensor_pattern,
        trend_summary=trend_summary
    )

    print("\n" + "=" * 60)
    print("PREDICTIVE HEALTH ANALYZER TEST")
    print("=" * 60)

    print("\nPredictive Health :",
          result["predictive_health"])

    print("Evidence Level    :",
          result["evidence_level"])

    print("Random Forest     :",
          result["random_forest"])

    print("LSTM              :",
          result["lstm"])

    print("Autoencoder       :",
          result["autoencoder"])

    print("Sensor Pattern    :",
          result["sensor_pattern"])

    print("Temporal Trend    :",
          result["temporal_trend"]["status"])

    print("\nIndicators:")

    for indicator in result["indicators"]:
        print(" -", indicator)

    print("\nRecommendation:")
    print(result["recommendation"])

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)