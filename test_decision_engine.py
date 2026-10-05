from decision_engine.decision_engine import DecisionEngine

engine = DecisionEngine()

engine.load_ai_outputs(
    health_status="Healthy",
    battery_prediction={
        "Battery_SOC": 85,
        "Battery_SOH": 96,
        "Battery_RUL": 1200
    },
    unknown_fault=False
)

decision = engine.evaluate_system_health()

print("\nReturned Decision:")
print(decision)

fault_decision = engine.evaluate_unknown_fault()

print("\nReturned Fault Decision:")
print(fault_decision)

soc_decision = engine.evaluate_battery_soc()

print("\nReturned SOC Decision:")
print(soc_decision)

soh_decision = engine.evaluate_battery_soh()

print("\nReturned SOH Decision:")
print(soh_decision)

rul_decision = engine.evaluate_battery_rul()

print("\nReturned RUL Decision:")
print(rul_decision)

overall = engine.evaluate_overall_system(
    decision,
    fault_decision,
    soc_decision,
    soh_decision,
    rul_decision
)

print("\nReturned Overall Decision:")
print(overall)

recommendation = engine.get_recommendation(
    overall["overall_status"]
)

print("\nReturned Recommendation:")
print(recommendation)

alert = engine.get_alert(
    overall["overall_status"]
)

print("\nReturned Alert:")
print(alert)