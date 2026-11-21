# =====================================================
# SolarSentinel-X
# Phase 13C : Real Sensor Data Preprocessing (Leak-Free)
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
# REAL SENSOR FEATURES (10-Feature Schema)
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
    print("SOLARSENTINEL-X REAL SENSOR PREPROCESSING (LEAK-FREE)")
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
        print("\nNo labelled real sensor data available.")
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

    print("\nOverall class distribution:")
    print(y.value_counts())

    # -------------------------------------------------
    # Train-Test Split (FIRST: Prevents Data Leakage)
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAIN-TEST SPLIT (LEAK-FREE)")
    print("=" * 70)

    class_counts = y.value_counts()

    if len(class_counts) >= 2 and class_counts.min() >= 2:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    else:
        print("\nNot enough labelled data for stratified split.")
        X_train_raw = X.copy()
        X_test_raw = X.copy()
        y_train = y.copy()
        y_test = y.copy()

    # -------------------------------------------------
    # Feature Scaling (Fit on Train ONLY)
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE SCALING (FIT ON TRAIN ONLY)")
    print("=" * 70)

    scaler = StandardScaler()

    # Fit ONLY on X_train to prevent any test distribution leakage
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    print("\nScaling completed successfully without data leakage.")
    print(f"X_train scaled shape : {X_train_scaled.shape}")
    print(f"X_test scaled shape  : {X_test_scaled.shape}")

    # -------------------------------------------------
    # Save processed data
    # -------------------------------------------------

    X_train_df = pd.DataFrame(
        X_train_scaled,
        columns=FEATURE_COLUMNS
    )

    X_test_df = pd.DataFrame(
        X_test_scaled,
        columns=FEATURE_COLUMNS
    )

    y_train_df = pd.DataFrame(
        y_train.reset_index(drop=True),
        columns=[TARGET_COLUMN]
    )

    y_test_df = pd.DataFrame(
        y_test.reset_index(drop=True),
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
    # Save leak-free scaler
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
    # Final Information & Class Distribution Breakdown
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)

    print(f"\nTraining samples : {len(X_train_df)}")
    print(f"Testing samples  : {len(X_test_df)}")

    print("\nTrain class distribution:")
    print(y_train_df[TARGET_COLUMN].value_counts())

    print("\nTest class distribution:")
    print(y_test_df[TARGET_COLUMN].value_counts())

    print("\nSaved files:")
    print("[OK] X_train_real.csv")
    print("[OK] X_test_real.csv")
    print("[OK] y_train_real.csv")
    print("[OK] y_test_real.csv")
    print("[OK] real_scaler.pkl (Fitted on train split only)")
    print("[OK] real_feature_names.csv")

    print(f"\nOutput folder:\n{OUTPUT_PATH}")


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    preprocess_real_dataset()