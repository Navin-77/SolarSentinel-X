"""
Alert Manager

Generates alerts based on the overall system status.
"""


def generate_alert(overall_status):
    """
    Generate alert message.
    """

    if overall_status == "HEALTHY":

        return (
            "INFO",
            "System operating normally."
        )

    elif overall_status == "WARNING":

        return (
            "WARNING",
            "Attention required. Check system health."
        )

    elif overall_status == "CRITICAL":

        return (
            "CRITICAL",
            "Immediate action required!"
        )

    return (
        "UNKNOWN",
        "Unknown system condition."
    )