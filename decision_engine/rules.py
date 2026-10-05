"""
Decision Rules

Contains all rule-based logic used by the Decision Engine.
"""


def evaluate_health_status(health_status):
    """
    Evaluate the current system health.
    """

    if health_status == "Healthy":

        return (
            "NORMAL",
            "System is operating normally."
        )

    elif health_status == "Warning":

        return (
            "WARNING",
            "System requires inspection."
        )

    elif health_status == "Critical":

        return (
            "CRITICAL",
            "Immediate maintenance required."
        )

    return (
        "UNKNOWN",
        "Unknown system status."
    )
    
def evaluate_unknown_fault(unknown_fault):
    """
    Evaluate unknown fault detected by the Autoencoder.
    """

    if unknown_fault:

        return (
            "ANOMALY",
            "Unknown fault detected. Immediate inspection required."
        )

    return (
        "NORMAL",
        "No unknown faults detected."
    )
    
def evaluate_battery_soc(battery_soc):
    """
    Evaluate Battery State of Charge (SOC).
    """

    if battery_soc >= 50:

        return (
            "NORMAL",
            "Battery charge level is sufficient."
        )

    elif battery_soc >= 20:

        return (
            "WARNING",
            "Battery charge is getting low."
        )

    return (
        "CRITICAL",
        "Battery charge is critically low. Recharge immediately."
    )
    
def evaluate_battery_soh(battery_soh):
    """
    Evaluate Battery State of Health (SOH).
    """

    if battery_soh >= 80:

        return (
            "NORMAL",
            "Battery health is good."
        )

    elif battery_soh >= 60:

        return (
            "WARNING",
            "Battery degradation detected."
        )

    return (
        "CRITICAL",
        "Battery replacement recommended."
    )
    
def evaluate_battery_rul(battery_rul):
    """
    Evaluate Battery Remaining Useful Life (RUL).
    """

    if battery_rul >= 1000:

        return (
            "NORMAL",
            "Battery remaining useful life is excellent."
        )

    elif battery_rul >= 500:

        return (
            "WARNING",
            "Battery is aging. Plan maintenance."
        )

    return (
        "CRITICAL",
        "Battery nearing end of life. Replacement recommended."
    )