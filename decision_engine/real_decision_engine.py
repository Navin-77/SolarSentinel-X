"""
SolarSentinel-X Real Decision Engine

Decision engine for the real ESP32 sensor + AI pipeline.

Uses:
    - Real Random Forest health prediction
    - Real Autoencoder anomaly detection
    - Real LSTM temporal health prediction

This module does NOT modify the existing Decision Engine.
"""


class RealDecisionEngine:

    def __init__(self):
        self.last_decision = None

    # ======================================================
    # HEALTH STATUS
    # ======================================================

    def evaluate_health(self, rf_health):
        """
        Evaluate the Random Forest health classification.
        """

        if rf_health == "Critical":
            return {
                "status": "CRITICAL",
                "message": "Immediate maintenance required."
            }

        if rf_health == "Warning":
            return {
                "status": "WARNING",
                "message": "System requires inspection."
            }

        return {
            "status": "NORMAL",
            "message": "System is operating normally."
        }

    # ======================================================
    # AUTOENCODER
    # ======================================================

    def evaluate_anomaly(self, anomaly_status):
        """
        Evaluate the real Autoencoder output.
        """

        if anomaly_status == "Anomaly":
            return {
                "status": "ANOMALY",
                "message": "Unknown sensor pattern detected. Inspection required."
            }

        return {
            "status": "NORMAL",
            "message": "No unknown sensor pattern detected."
        }

    # ======================================================
    # LSTM
    # ======================================================

    def evaluate_temporal_health(self, lstm_health):
        """
        Evaluate the real LSTM temporal health prediction.

        If fewer than 10 samples have been received, the LSTM
        result is unavailable and is not used to force a decision.
        """

        if lstm_health is None:
            return {
                "status": "UNAVAILABLE",
                "message": "Waiting for sufficient temporal sensor data."
            }

        if lstm_health == "Critical":
            return {
                "status": "CRITICAL",
                "message": "Temporal model indicates critical system health."
            }

        if lstm_health == "Warning":
            return {
                "status": "WARNING",
                "message": "Temporal model indicates warning-level system health."
            }

        return {
            "status": "NORMAL",
            "message": "Temporal model indicates normal system health."
        }

    # ======================================================
    # OVERALL STATUS
    # ======================================================

    def evaluate_overall_status(
        self,
        health_result,
        anomaly_result,
        temporal_result
    ):
        """
        Combine the three real AI outputs.

        Priority:
            CRITICAL > ANOMALY/WARNING > NORMAL

        LSTM UNAVAILABLE is ignored until enough samples
        are available.
        """

        statuses = [
            health_result["status"],
            anomaly_result["status"],
            temporal_result["status"]
        ]

        if "CRITICAL" in statuses:
            return "CRITICAL"

        if "ANOMALY" in statuses:
            return "WARNING"

        if "WARNING" in statuses:
            return "WARNING"

        return "NORMAL"

    # ======================================================
    # RECOMMENDATION
    # ======================================================

    def generate_recommendation(self, overall_status):
        """
        Generate an operational recommendation.
        """

        if overall_status == "CRITICAL":
            return (
                "Immediate inspection and maintenance are recommended."
            )

        if overall_status == "WARNING":
            return (
                "Inspect the system and monitor sensor conditions."
            )

        return (
            "System is operating normally. Continue monitoring."
        )

    # ======================================================
    # ALERT
    # ======================================================

    def generate_alert(self, overall_status):

        if overall_status == "CRITICAL":
            return (
                "CRITICAL ALERT: Immediate maintenance required."
            )

        if overall_status == "WARNING":
            return (
                "WARNING: System requires inspection."
            )

        return (
            "NORMAL: No immediate action required."
        )

    # ======================================================
    # COMPLETE DECISION
    # ======================================================

    def evaluate(
        self,
        rf_health,
        anomaly_status,
        lstm_health=None
    ):
        """
        Run the complete real decision process.
        """

        health_result = self.evaluate_health(
            rf_health
        )

        anomaly_result = self.evaluate_anomaly(
            anomaly_status
        )

        temporal_result = self.evaluate_temporal_health(
            lstm_health
        )

        overall_status = self.evaluate_overall_status(
            health_result,
            anomaly_result,
            temporal_result
        )

        recommendation = self.generate_recommendation(
            overall_status
        )

        alert = self.generate_alert(
            overall_status
        )

        result = {
            "health": health_result,
            "anomaly": anomaly_result,
            "temporal_health": temporal_result,
            "overall_status": overall_status,
            "recommendation": recommendation,
            "alert": alert
        }

        self.last_decision = result

        return result


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    engine = RealDecisionEngine()

    print("=" * 60)
    print("SolarSentinel-X Real Decision Engine Test")
    print("=" * 60)

    # Test 1
    print("\nTEST 1 — Healthy")

    result = engine.evaluate(
        rf_health="Healthy",
        anomaly_status="Normal",
        lstm_health="Healthy"
    )

    print("Overall Status :", result["overall_status"])
    print("Recommendation :", result["recommendation"])
    print("Alert          :", result["alert"])

    # Test 2
    print("\nTEST 2 — Warning")

    result = engine.evaluate(
        rf_health="Warning",
        anomaly_status="Normal",
        lstm_health="Warning"
    )

    print("Overall Status :", result["overall_status"])
    print("Recommendation :", result["recommendation"])
    print("Alert          :", result["alert"])

    # Test 3
    print("\nTEST 3 — Critical")

    result = engine.evaluate(
        rf_health="Critical",
        anomaly_status="Normal",
        lstm_health="Critical"
    )

    print("Overall Status :", result["overall_status"])
    print("Recommendation :", result["recommendation"])
    print("Alert          :", result["alert"])

    # Test 4
    print("\nTEST 4 — Unknown Anomaly")

    result = engine.evaluate(
        rf_health="Healthy",
        anomaly_status="Anomaly",
        lstm_health="Healthy"
    )

    print("Overall Status :", result["overall_status"])
    print("Recommendation :", result["recommendation"])
    print("Alert          :", result["alert"])

    print("\nReal Decision Engine test completed.")