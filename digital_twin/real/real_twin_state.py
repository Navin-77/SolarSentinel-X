"""
SolarSentinel-X Real Digital Twin State

Stores the current real sensor state and AI predictions.
This module is designed for the 10-feature ESP32 real-sensor pipeline.
"""


class RealTwinState:
    """
    Maintains the current state of the real solar microgrid
    digital twin.
    """

    def __init__(self):
        # Latest real sensor measurements
        self.sensor_data = {}

        # Random Forest prediction
        self.rf_prediction = None
        self.rf_confidence = None
        self.rf_probabilities = None

        # Autoencoder prediction
        self.anomaly_status = None
        self.reconstruction_error = None
        self.anomaly_threshold = None

        # LSTM prediction
        self.lstm_prediction = None
        self.lstm_confidence = None
        self.lstm_probabilities = None

        # SHAP explanation
        self.shap_explanation = None

        # Overall digital twin state
        self.overall_status = None

        # Decision engine output
        self.recommendation = None
        self.alert = None
        
        # Predictive Health Analyzer output
        self.predictive_health = None
        self.predictive_evidence_level = None
        self.predictive_indicators = []
        self.predictive_recommendation = None

        # Timestamp
        self.timestamp = None

    def update_sensor_data(self, sensor_data):
        """
        Update the latest real sensor measurements.
        """
        self.sensor_data = sensor_data.copy()

    def update_rf(self, prediction, confidence=None, probabilities=None):
        """
        Update Random Forest results.
        """
        self.rf_prediction = prediction
        self.rf_confidence = confidence
        self.rf_probabilities = probabilities

    def update_autoencoder(
        self,
        anomaly_status,
        reconstruction_error=None,
        threshold=None
    ):
        """
        Update Autoencoder anomaly detection results.
        """
        self.anomaly_status = anomaly_status
        self.reconstruction_error = reconstruction_error
        self.anomaly_threshold = threshold

    def update_lstm(self, prediction, confidence=None, probabilities=None):
        """
        Update LSTM temporal prediction results.
        """
        self.lstm_prediction = prediction
        self.lstm_confidence = confidence
        self.lstm_probabilities = probabilities

    def update_shap(self, explanation):
        """
        Update SHAP explanation.
        """
        self.shap_explanation = explanation

    def update_decision(self, overall_status, recommendation, alert):
        """
        Update final decision-engine output.
        """
        self.overall_status = overall_status
        self.recommendation = recommendation
        self.alert = alert
        
    def update_predictive_health(
        self,
        predictive_health_result
    ):
        """
        Update Predictive Health Analyzer output.
        """

        if predictive_health_result is None:
            return

        self.predictive_health = (
            predictive_health_result.get(
                "predictive_health"
            )
        )

        self.predictive_evidence_level = (
            predictive_health_result.get(
                "evidence_level"
            )
        )

        self.predictive_indicators = (
            predictive_health_result.get(
                "indicators",
                []
            )
        )

        self.predictive_recommendation = (
            predictive_health_result.get(
                "recommendation"
            )
        )    

    def update_timestamp(self, timestamp):
        """
        Update the timestamp associated with the current state.
        """
        self.timestamp = timestamp

    def get_state(self):
        """
        Return the complete current digital twin state.
        """
        return {
            "timestamp": self.timestamp,
            "sensor_data": self.sensor_data,
            "random_forest": {
                "prediction": self.rf_prediction,
                "confidence": self.rf_confidence,
                "probabilities": self.rf_probabilities,
            },
            "autoencoder": {
                "status": self.anomaly_status,
                "reconstruction_error": self.reconstruction_error,
                "threshold": self.anomaly_threshold,
            },
            "lstm": {
                "prediction": self.lstm_prediction,
                "confidence": self.lstm_confidence,
                "probabilities": self.lstm_probabilities,
            },
            "shap": self.shap_explanation,
            "overall_status": self.overall_status,
            "recommendation": self.recommendation,
            "alert": self.alert,
            
            "predictive_health": self.predictive_health,
            "predictive_evidence_level": self.predictive_evidence_level,
            "predictive_indicators": self.predictive_indicators,
            "predictive_recommendation": self.predictive_recommendation,
        }