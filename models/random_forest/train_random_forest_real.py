# =====================================================
# SolarSentinel-X
# Phase 14 : Real Sensor Random Forest
# =====================================================

from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "real_training"
    / "real_sensor_dataset.csv"
)

MODEL_FOLDER = (
    BASE_DIR
    / "models"
    / "random_forest"
    / "real_trained"
)

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------------------------------
# 10 REAL SENSOR FEATURES
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
# MAIN
# =====================================================

def train_real_random_forest():

    print("=" * 70)
    print("SOLARSENTINEL-X REAL SENSOR RANDOM FOREST")
    print("=" * 70)

    # -------------------------------------------------
    # Check dataset
    # -------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"\nReal dataset not found:\n{DATASET_PATH}"
        )

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset loaded successfully!")

    print(f"Total rows    : {len(df)}")
    print(f"Total columns : {len(df.columns)}")

    # -------------------------------------------------
    # Remove Unknown labels
    # -------------------------------------------------

    df = df[
        df[TARGET_COLUMN].notna()
        & (
            df[TARGET_COLUMN]
            .astype(str)
            .str.lower()
            != "unknown"
        )
    ].copy()

    print(f"\nLabelled rows : {len(df)}")

    # -------------------------------------------------
    # Safety check
    # -------------------------------------------------

    if len(df) < 30:

        print("\n" + "=" * 70)
        print("NOT ENOUGH REAL DATA")
        print("=" * 70)

        print(
            "\nAt least 30 labelled real sensor samples "
            "are required before training."
        )

        print(
            "\nCurrent labelled samples:",
            len(df)
        )

        print(
            "\nCollect more REAL ESP32 data first."
        )

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
            f"\nMissing features: {missing_features}"
        )

    # -------------------------------------------------
    # Feature matrix
    # -------------------------------------------------

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMN].copy()

    # Convert features to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove invalid rows
    valid_rows = X.notna().all(axis=1)

    X = X.loc[valid_rows].reset_index(
        drop=True
    )

    y = y.loc[valid_rows].reset_index(
        drop=True
    )

    print(
        f"\nValid training rows : {len(X)}"
    )

    # -------------------------------------------------
    # Display class distribution
    # -------------------------------------------------

    print("\nSystem Health Distribution:")

    print(
        y.value_counts()
    )

    # -------------------------------------------------
    # Encode target
    # -------------------------------------------------

    health_encoder = LabelEncoder()

    y_encoded = health_encoder.fit_transform(y)

    print("\nHealth Encoding:")

    for index, label in enumerate(
        health_encoder.classes_
    ):

        print(
            f"{label} --> {index}"
        )

    # -------------------------------------------------
    # Feature Scaling
    # -------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -------------------------------------------------
    # Train/Test Split
    # -------------------------------------------------

    class_counts = y.value_counts()

    if (
        len(class_counts) >= 2
        and class_counts.min() >= 2
    ):

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X_scaled,
                y_encoded,
                test_size=0.20,
                random_state=42,
                stratify=y_encoded
            )
        )

    else:

        print(
            "\nERROR: At least two classes with "
            "multiple samples are required."
        )

        return

    print("\nDataset Split:")

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    # -------------------------------------------------
    # Random Forest
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nRandom Forest training completed!"
    )

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    # -------------------------------------------------
    # Classification Report
    # -------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=health_encoder.classes_,
            zero_division=0
        )
    )

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # -------------------------------------------------
    # Feature Importance
    # -------------------------------------------------

    print("\nFeature Importance:")

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS
    ).sort_values(
        ascending=False
    )

    print(importance)

    # -------------------------------------------------
    # Save model
    # -------------------------------------------------

    model_path = (
        MODEL_FOLDER
        / "random_forest_real.pkl"
    )

    scaler_path = (
        MODEL_FOLDER
        / "real_scaler.pkl"
    )

    encoder_path = (
        MODEL_FOLDER
        / "health_encoder_real.pkl"
    )

    feature_path = (
        MODEL_FOLDER
        / "feature_names_real.csv"
    )

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    joblib.dump(
        health_encoder,
        encoder_path
    )

    pd.DataFrame({
        "Feature": FEATURE_COLUMNS
    }).to_csv(
        feature_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"\n✓ {model_path}"
    )

    print(
        f"✓ {scaler_path}"
    )

    print(
        f"✓ {encoder_path}"
    )

    print(
        f"✓ {feature_path}"
    )


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    train_real_random_forest()