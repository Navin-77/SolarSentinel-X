from digital_twin.digital_twin import DigitalTwin

twin = DigitalTwin()

# ------------------------------------
# Current Health Prediction
# ------------------------------------

health = twin.predict_current_health()

print("\nReturned Health Status:")
print(health)

# ------------------------------------
# Future Battery Prediction
# ------------------------------------

battery_prediction = twin.predict_future_battery()

print("\nReturned Battery Prediction:")
print(battery_prediction)

# ------------------------------------
# Unknown Fault Detection
# ------------------------------------

fault = twin.detect_unknown_fault()

print("\nUnknown Fault Detection:")
print(fault)

# ------------------------------------
# SHAP Explainability
# ------------------------------------

explanation = twin.explain_prediction()

print("\nTop Feature Importance:")
print(explanation[:5])