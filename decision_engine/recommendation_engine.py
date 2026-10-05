"""
Recommendation Engine

Generates maintenance recommendations based on
Decision Engine outputs.
"""

def generate_recommendation(overall_status):
    """
    Generate recommendations based on the overall system status.
    """

    if overall_status == "HEALTHY":

        return (
            "Continue normal system operation."
        )

    elif overall_status == "WARNING":

        return (
            "Schedule preventive maintenance and monitor the system."
        )

    elif overall_status == "CRITICAL":

        return (
            "Stop system operation immediately and perform maintenance."
        )

    return (
        "Investigate system status."
    )