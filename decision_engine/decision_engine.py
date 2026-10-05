"""
Decision Engine

Coordinates all rule-based decisions for SolarSentinel-X.
"""
from decision_engine.rules import (
    evaluate_health_status,
    evaluate_unknown_fault,
    evaluate_battery_soc,
    evaluate_battery_soh,
    evaluate_battery_rul,
)

from decision_engine.recommendation_engine import generate_recommendation

from decision_engine.alert_manager import generate_alert

class DecisionEngine:
    """
    Main Rule-Based Decision Engine.
    """

    def __init__(self):

        print("===================================")
        print(" Decision Engine Initialized")
        print("===================================")

    def load_ai_outputs(
        self,
        health_status,
        battery_prediction,
        unknown_fault
    ):
        """
        Load AI outputs from the Digital Twin.
        """

        self.health_status = health_status

        self.battery_prediction = battery_prediction

        self.unknown_fault = unknown_fault

        print("\n========== AI Outputs Loaded ==========")
        print(f"Health Status : {self.health_status}")
        print(f"Battery Prediction : {self.battery_prediction}")
        print(f"Unknown Fault : {self.unknown_fault}")
        print("=======================================")
        
    def evaluate_system_health(self):
        """
        Evaluate health using rule-based logic.
        """

        status, message = evaluate_health_status(
            self.health_status
        )

        print("\n========== Health Decision ==========")
        print(f"Decision Level : {status}")
        print(f"Message        : {message}")
        print("=====================================")

        return {
            "status": status,
            "message": message
        }
        
    def evaluate_unknown_fault(self):
        """
        Evaluate unknown faults using rule-based logic.
        """

        status, message = evaluate_unknown_fault(
            self.unknown_fault
        )

        print("\n========== Unknown Fault Decision ==========")
        print(f"Decision Level : {status}")
        print(f"Message        : {message}")
        print("============================================")

        return {
            "status": status,
            "message": message
        }
        
    def evaluate_battery_soc(self):
        """
        Evaluate Battery SOC using rule-based logic.
        """

        battery_soc = self.battery_prediction["Battery_SOC"]

        status, message = evaluate_battery_soc(
            battery_soc
        )

        print("\n========== Battery SOC Decision ==========")
        print(f"Battery SOC   : {battery_soc}%")
        print(f"Decision Level: {status}")
        print(f"Message       : {message}")
        print("==========================================")

        return {
            "status": status,
            "message": message
        }
        
    def evaluate_battery_soh(self):
        """
        Evaluate Battery SOH using rule-based logic.
        """

        battery_soh = self.battery_prediction["Battery_SOH"]

        status, message = evaluate_battery_soh(
            battery_soh
        )

        print("\n========== Battery SOH Decision ==========")
        print(f"Battery SOH   : {battery_soh}%")
        print(f"Decision Level: {status}")
        print(f"Message       : {message}")
        print("==========================================")

        return {
            "status": status,
            "message": message
        }
        
    def evaluate_battery_rul(self):
        """
        Evaluate Battery Remaining Useful Life (RUL).
        """

        battery_rul = self.battery_prediction["Battery_RUL"]

        status, message = evaluate_battery_rul(
            battery_rul
        )

        print("\n========== Battery RUL Decision ==========")
        print(f"Battery RUL   : {battery_rul}")
        print(f"Decision Level: {status}")
        print(f"Message       : {message}")
        print("==========================================")

        return {
            "status": status,
            "message": message
        }
        

    def evaluate_overall_system(
        self,
        health,
        fault,
        soc,
        soh,
        rul
    ):
        """
        Combine all rule-based decisions into one overall system status.
        """

        decisions = [
            health["status"],
            fault["status"],
            soc["status"],
            soh["status"],
            rul["status"]
        ]

        if "CRITICAL" in decisions:
            overall = "CRITICAL"

        elif "WARNING" in decisions or "ANOMALY" in decisions:
            overall = "WARNING"

        else:
            overall = "HEALTHY"

        print("\n========== Overall System Status ==========")
        print(f"Overall Status : {overall}")
        print("===========================================")

        return {
            "overall_status": overall,
            "individual_decisions": decisions
        }
        
    def get_recommendation(self, overall_status):
        """
        Generate maintenance recommendation.
        """

        recommendation = generate_recommendation(
            overall_status
        )

        print("\n========== Recommendation ==========")
        print(recommendation)
        print("====================================")

        return recommendation
    
    def get_alert(self, overall_status):
        """
        Generate alert message.
        """

        level, message = generate_alert(
            overall_status
        )

        print("\n========== Alert ==========")
        print(f"Alert Level : {level}")
        print(f"Message     : {message}")
        print("===========================")

        return {
            "level": level,
            "message": message
        }