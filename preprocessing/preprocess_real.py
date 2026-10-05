# =====================================================
# SolarSentinel-X
# Phase 13C : Real Sensor Data Preprocessing
# =====================================================

from pathlib import Path

import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "real_training"
    / "real_sensor_dataset.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "dataset"
    / "real_processed"
)

OUTPUT_PATH.mkdir(exist_ok=True)


# -----------------------------------------------------
# REAL SENSOR FEATURES
# -----------------------------------------------------

FEATURE_COLUMNS = [
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


TARGET_COLUMN = "System_Health"


# =====================================================
# MAIN PREPROCESSING FUNCTION
# =====================================================

def preprocess_real_dataset():

    print("=" * 70)
    print("SOLARSENTINEL-X REAL SENSOR PREPROCESSING")
    print("=" * 70)

    # -------------------------------------------------
    # Check dataset
    # -------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset Loaded Successfully!")

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # -------------------------------------------------
    # Remove invalid / test rows
    # -------------------------------------------------

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    # Remove rows whose label is Unknown
    df = df[
        df[TARGET_COLUMN].notna()
        & (df[TARGET_COLUMN].astype(str).str.lower() != "unknown")
    ].copy()

    print(f"\nUsable labelled rows : {len(df)}")

    if len(df) == 0:
        print("\nNo labelled real sensor data available yet.")
        print("Preprocessing will begin after real labelled data")
        print("has been collected from the ESP32.")
        return

    # -------------------------------------------------
    # Verify features
    # -------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # -------------------------------------------------
    # Select features and target
    # -------------------------------------------------

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    print("\n" + "=" * 70)
    print("10 REAL SENSOR FEATURES")
    print("=" * 70)

    for index, feature in enumerate(FEATURE_COLUMNS, start=1):
        print(f"{index:2d}. {feature}")

    print(f"\nTarget : {TARGET_COLUMN}")

    # -------------------------------------------------
    # Convert feature values to numeric
    # -------------------------------------------------

    X = X.apply(pd.to_numeric, errors="coerce")

    # Remove invalid rows
    valid_rows = X.notna().all(axis=1)

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    print(f"\nValid numeric rows : {len(X)}")

    if len(X) == 0:
        print("\nNo valid numeric rows available.")
        return

    # -------------------------------------------------
    # Feature Scaling
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE SCALING")
    print("=" * 70)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("\nScaling completed.")

    print(f"Feature matrix shape : {X_scaled.shape}")

    # -------------------------------------------------
    # Train-Test Split
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAIN-TEST SPLIT")
    print("=" * 70)

    # Stratification is used only when every class has
    # enough samples for a valid split.

    class_counts = y.value_counts()

    if len(class_counts) >= 2 and class_counts.min() >= 2:

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

    else:

        print(
            "\nNot enough labelled data for stratified split."
        )

        print(
            "Saving the processed dataset without train/test split."
        )

        X_train = X_scaled
        X_test = X_scaled
        y_train = y
        y_test = y

    # -------------------------------------------------
    # Save processed data
    # -------------------------------------------------

    X_train_df = pd.DataFrame(
        X_train,
        columns=FEATURE_COLUMNS
    )

    X_test_df = pd.DataFrame(
        X_test,
        columns=FEATURE_COLUMNS
    )

    y_train_df = pd.DataFrame(
        y_train,
        columns=[TARGET_COLUMN]
    )

    y_test_df = pd.DataFrame(
        y_test,
        columns=[TARGET_COLUMN]
    )

    X_train_df.to_csv(
        OUTPUT_PATH / "X_train_real.csv",
        index=False
    )

    X_test_df.to_csv(
        OUTPUT_PATH / "X_test_real.csv",
        index=False
    )

    y_train_df.to_csv(
        OUTPUT_PATH / "y_train_real.csv",
        index=False
    )

    y_test_df.to_csv(
        OUTPUT_PATH / "y_test_real.csv",
        index=False
    )

    # -------------------------------------------------
    # Save scaler
    # -------------------------------------------------

    joblib.dump(
        scaler,
        OUTPUT_PATH / "real_scaler.pkl"
    )

    # -------------------------------------------------
    # Save feature names
    # -------------------------------------------------

    pd.DataFrame({
        "Feature": FEATURE_COLUMNS
    }).to_csv(
        OUTPUT_PATH / "real_feature_names.csv",
        index=False
    )

    # -------------------------------------------------
    # Final information
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)

    print(f"\nTraining samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    print("\nSaved files:")

    print("✓ X_train_real.csv")
    print("✓ X_test_real.csv")
    print("✓ y_train_real.csv")
    print("✓ y_test_real.csv")
    print("✓ real_scaler.pkl")
    print("✓ real_feature_names.csv")

    print(f"\nOutput folder:")
    print(OUTPUT_PATH)


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    preprocess_real_dataset()