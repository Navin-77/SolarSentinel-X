"""
Prediction Logger

Stores every AI prediction into a CSV file.
"""

from pathlib import Path
import pandas as pd

LOG_FOLDER = Path(__file__).resolve().parent.parent / "logs"
LOG_FOLDER.mkdir(exist_ok=True)

LOG_FILE = LOG_FOLDER / "prediction_log.csv"


def log_prediction(data):
    """
    Append one prediction row to the CSV log.
    Avoid writing duplicate consecutive entries.
    """

    row = pd.DataFrame([{

        "Date": data["last_updated_date"],
        "Time": data["last_updated_time"],

        "Health": data["health"],

        "Battery_SOC": data["soc"],
        "Battery_SOH": data["soh"],
        "Battery_RUL": data["rul"],

        "Unknown_Fault": data["unknown_fault"],

        "Overall_Status": data["overall_status"]

    }])

    # First entry
    if not LOG_FILE.exists():
        row.to_csv(LOG_FILE, index=False)
        return

    # Read last logged row
    existing = pd.read_csv(LOG_FILE)

    if not existing.empty:

        last_row = existing.iloc[-1]

        is_duplicate = (

            last_row["Date"] == row.iloc[0]["Date"] and
            last_row["Time"] == row.iloc[0]["Time"] and
            last_row["Health"] == row.iloc[0]["Health"] and
            last_row["Battery_SOC"] == row.iloc[0]["Battery_SOC"] and
            last_row["Battery_SOH"] == row.iloc[0]["Battery_SOH"] and
            last_row["Battery_RUL"] == row.iloc[0]["Battery_RUL"] and
            last_row["Unknown_Fault"] == row.iloc[0]["Unknown_Fault"] and
            last_row["Overall_Status"] == row.iloc[0]["Overall_Status"]

        )

        if is_duplicate:
            return

    row.to_csv(
        LOG_FILE,
        mode="a",
        header=False,
        index=False
    )