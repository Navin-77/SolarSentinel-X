"""
SolarSentinel-X — Dynamic Real Dataset Profile

Reads the current real sensor CSV directly.

IMPORTANT:
- No sensor thresholds are hard-coded here.
- Replace the CSV tomorrow and restart the application.
- The profile is recalculated from the new CSV.
- These values are ONLY for Digital Twin visualization.
- They are NOT AI predictions.
"""

from pathlib import Path
import math
import pandas as pd


# ============================================================
# REQUIRED FEATURES
# ============================================================

FEATURES = [
    "LDR",
    "DHT22_Temperature",
    "DHT22_Humidity",
    "DS18B20_Temperature",
    "Solar_Voltage",
    "Solar_Current",
    "Solar_Power",
    "Battery_Voltage",
    "Battery_Current",
    "Battery_Power",
]

LABELS = [
    "Healthy",
    "Warning",
    "Critical",
]


# ============================================================
# DATASET LOCATION
# ============================================================

CSV_PATH = Path(
    "dataset/real_training/real_sensor_dataset.csv"
)


# ============================================================
# FIND DATASET
# ============================================================

def find_real_dataset():
    """
    Find the current real dataset.
    """

    if CSV_PATH.exists():
        return CSV_PATH

    # Fallback if running from another directory.
    alternatives = [
        Path("real_sensor_dataset.csv"),
        Path("../dataset/real_training/real_sensor_dataset.csv"),
    ]

    for path in alternatives:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Real sensor dataset not found.\n"
        "Expected location:\n"
        "dataset/real_training/real_sensor_dataset.csv"
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_real_dataset():
    """
    Load and validate the current real CSV.
    """

    path = find_real_dataset()

    df = pd.read_csv(path)

    required_columns = FEATURES + ["System_Health"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns:\n"
            f"{missing_columns}"
        )

    # Convert all sensor values to numeric.
    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # Standardize labels.
    df["System_Health"] = (
        df["System_Health"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # Remove unusable rows.
    df = df.dropna(
        subset=FEATURES + ["System_Health"]
    )

    return df, path


# ============================================================
# MIDPOINT
# ============================================================

def _midpoint(a, b):
    return (float(a) + float(b)) / 2.0


# ============================================================
# BUILD DATASET PROFILE
# ============================================================

def build_profile(df):
    """
    Calculate statistics directly from the current CSV.

    The class medians are used as representative points for
    Digital Twin visualization.

    This does NOT replace the trained AI models.
    """

    profile = {}

    for feature in FEATURES:

        class_statistics = {}

        for label in LABELS:

            values = df.loc[
                df["System_Health"] == label,
                feature
            ].dropna()

            if values.empty:

                class_statistics[label] = {
                    "count": 0,
                    "min": None,
                    "max": None,
                    "median": None,
                }

            else:

                class_statistics[label] = {
                    "count": int(values.count()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median": float(values.median()),
                }

        # Get available class medians.
        available = [
            (
                label,
                class_statistics[label]["median"]
            )
            for label in LABELS
            if class_statistics[label]["median"] is not None
        ]

        # Sort by measured value.
        ordered = sorted(
            available,
            key=lambda item: item[1]
        )

        # Calculate boundaries between class centres.
        boundaries = []

        for i in range(len(ordered) - 1):

            boundaries.append(
                _midpoint(
                    ordered[i][1],
                    ordered[i + 1][1]
                )
            )

        profile[feature] = {

            "classes": class_statistics,

            "ordered_medians": ordered,

            "boundaries": boundaries,
        }

    return profile


# ============================================================
# LOAD CURRENT PROFILE
# ============================================================

def load_profile():

    df, dataset_path = load_real_dataset()

    profile = build_profile(df)

    return {

        "csv_path": str(dataset_path),

        "rows": int(len(df)),

        "class_counts": (
            df["System_Health"]
            .value_counts()
            .to_dict()
        ),

        "features": FEATURES,

        "profile": profile,
    }


# ============================================================
# CLASSIFY LIVE VALUE FOR VISUALIZATION
# ============================================================

def classify_visual(
    value,
    feature,
    profile_data
):
    """
    Convert a live measurement into a dataset-relative
    visual state.

    IMPORTANT:
    This is NOT an AI health diagnosis.

    RF/LSTM/Autoencoder remain responsible for AI inference.
    """

    try:

        value = float(value)

    except (TypeError, ValueError):

        return "UNAVAILABLE"

    if not math.isfinite(value):

        return "UNAVAILABLE"

    feature_profile = profile_data[
        "profile"
    ].get(feature)

    if not feature_profile:

        return "NORMAL"

    boundaries = feature_profile[
        "boundaries"
    ]

    if not boundaries:

        return "NORMAL"

    # Two class centres.
    if len(boundaries) == 1:

        if value < boundaries[0]:

            return "LOW"

        return "HIGH"

    # Three class centres.
    low_boundary = boundaries[0]

    high_boundary = boundaries[-1]

    if value < low_boundary:

        return "LOW"

    if value > high_boundary:

        return "HIGH"

    return "NORMAL"


# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary(profile_data):

    return (

        f"Dataset: "
        f"{profile_data['csv_path']} | "

        f"Rows: "
        f"{profile_data['rows']} | "

        f"Classes: "
        f"{profile_data['class_counts']}"

    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "SolarSentinel-X — Dynamic Dataset Profile"
    )

    print("=" * 70)

    data = load_profile()

    print()

    print(
        dataset_summary(data)
    )

    print()

    print("Feature profiles:")

    for feature in FEATURES:

        info = data["profile"][feature]

        print()

        print(feature)

        for label in LABELS:

            stats = info["classes"][label]

            print(
                f"  {label:<10} "
                f"count={stats['count']:<4} "
                f"min={stats['min']} "
                f"max={stats['max']} "
                f"median={stats['median']}"
            )

    print()

    print("=" * 70)

    print(
        "Dataset profile loaded successfully."
    )

    print("=" * 70)