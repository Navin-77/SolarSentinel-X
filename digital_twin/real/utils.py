"""
Utility functions for the Digital Twin.
"""

from pathlib import Path
import pandas as pd


def load_latest_sensor_data():
    """
    Load the latest processed sensor data row.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame containing the latest sensor values.
    """

    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "dataset"
        / "processed"
        / "X_test.csv"
    )

    df = pd.read_csv(data_path)

    latest_sequence = df.tail(24).reset_index(drop=True)

    return latest_sequence

def load_latest_dashboard_data():
    """
    Load the latest engineering sensor values for dashboard display.
    """

    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "dataset"
        / "synthetic"
        / "solar_microgrid_dataset.csv"
    )

    df = pd.read_csv(data_path)

    return df.iloc[-1]

def load_dashboard_history():
    """
    Load the latest 24 records for dashboard visualization.
    """

    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "dataset"
        / "synthetic"
        / "solar_microgrid_dataset.csv"
    )

    df = pd.read_csv(data_path)

    return df.tail(24).reset_index(drop=True)

