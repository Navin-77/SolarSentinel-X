from digital_twin.digital_twin import DigitalTwin
from decision_engine.decision_engine import DecisionEngine

print("=" * 60)
print(" SolarSentinel-X Complete AI Pipeline ")
print("=" * 60)

# -----------------------------------------
# Initialize Modules
# -----------------------------------------

digital_twin = DigitalTwin()
decision_engine = DecisionEngine()

# -----------------------------------------
# Random Forest Prediction
# -----------------------------------------

health_status = digital_twin.predict_current_health()

print("\nReturned Health Status:")
print(health_status)

# -----------------------------------------
# LSTM Prediction
# -----------------------------------------

battery_prediction = digital_twin.predict_future_battery()

print("\nReturned Battery Prediction:")
print(battery_prediction)

# -----------------------------------------
# Autoencoder Prediction
# -----------------------------------------

unknown_fault = digital_twin.detect_unknown_fault()

print("\nReturned Unknown Fault:")
print(unknown_fault)

print("\nReturned Unknown Fault:")
print(unknown_fault)

# -----------------------------------------
# Load AI outputs into Decision Engine
# -----------------------------------------

decision_engine.load_ai_outputs(
    health_status=health_status,
    battery_prediction=battery_prediction,
    unknown_fault=unknown_fault["unknown_fault"]
)

print("\nAI outputs successfully passed to Decision Engine.")

# -----------------------------------------
# Decision Engine Evaluation
# -----------------------------------------

health_decision = decision_engine.evaluate_system_health()

fault_decision = decision_engine.evaluate_unknown_fault()

soc_decision = decision_engine.evaluate_battery_soc()

soh_decision = decision_engine.evaluate_battery_soh()

rul_decision = decision_engine.evaluate_battery_rul()

overall = decision_engine.evaluate_overall_system(
    health_decision,
    fault_decision,
    soc_decision,
    soh_decision,
    rul_decision
)

recommendation = decision_engine.get_recommendation(
    overall["overall_status"]
)

alert = decision_engine.get_alert(
    overall["overall_status"]
)