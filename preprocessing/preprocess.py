# =====================================================
# SolarSentinel-X
# Phase 2 : Data Preprocessing
# =====================================================

from pathlib import Path

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "synthetic"
    / "solar_microgrid_dataset.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
)

OUTPUT_PATH.mkdir(exist_ok=True)

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

print("=" * 70)
print("SOLARSENTINEL-X DATA PREPROCESSING")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print("\nDataset Loaded Successfully!")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# =====================================================
# Feature Selection
# =====================================================

print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

# Features
X = df.drop(
    columns=[
        "Timestamp",
        "Fault_Type",
        "Health_Score",
        "System_Health"
    ]
)

# Target
y = df["System_Health"]

print(f"\nNumber of Features : {X.shape[1]}")
print(f"Target Variable    : {y.name}")

print("\nFeature Names:\n")

for column in X.columns:
    print(column)
    
# =====================================================
# Label Encoding
# =====================================================

print("\n" + "=" * 70)
print("LABEL ENCODING")
print("=" * 70)

# Create Label Encoders
charging_encoder = LabelEncoder()
health_encoder = LabelEncoder()

# Encode Charging Status
X["Charging_Status"] = charging_encoder.fit_transform(
    X["Charging_Status"]
)

# Encode Target Variable
y = health_encoder.fit_transform(y)

print("\nCharging_Status Encoding:")

for index, label in enumerate(charging_encoder.classes_):
    print(f"{label} --> {index}")

print("\nSystem_Health Encoding:")

for index, label in enumerate(health_encoder.classes_):
    print(f"{label} --> {index}")
    
# =====================================================
# Feature Scaling
# =====================================================

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)

# Create StandardScaler
scaler = StandardScaler()

# Scale all feature columns
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling completed successfully!")

print(f"Scaled Feature Matrix Shape : {X_scaled.shape}")

print("\nFirst Scaled Sample:")

print(X_scaled[0])

# =====================================================
# Train-Test Split
# =====================================================

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset Split Completed Successfully!")

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

print(f"\nTraining Feature Shape : {X_train.shape}")
print(f"Testing Feature Shape  : {X_test.shape}")

print(f"\nTraining Labels Shape : {y_train.shape}")
print(f"Testing Labels Shape  : {y_test.shape}")

# =====================================================
# Save Processed Dataset
# =====================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED DATASETS")
print("=" * 70)

# Convert NumPy arrays back to DataFrames
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

y_train_df = pd.DataFrame(y_train, columns=["System_Health"])
y_test_df = pd.DataFrame(y_test, columns=["System_Health"])

# Save CSV files
X_train_df.to_csv(OUTPUT_PATH / "X_train.csv", index=False)
X_test_df.to_csv(OUTPUT_PATH / "X_test.csv", index=False)

y_train_df.to_csv(OUTPUT_PATH / "y_train.csv", index=False)
y_test_df.to_csv(OUTPUT_PATH / "y_test.csv", index=False)

print("\nProcessed datasets saved successfully!")

print(f"\nLocation: {OUTPUT_PATH}")

print("\nSaved Files:")
print("✓ X_train.csv")
print("✓ X_test.csv")
print("✓ y_train.csv")
print("✓ y_test.csv")

# =====================================================
# Save Preprocessing Objects
# =====================================================

joblib.dump(scaler, OUTPUT_PATH / "scaler.pkl")
joblib.dump(charging_encoder, OUTPUT_PATH / "charging_encoder.pkl")
joblib.dump(health_encoder, OUTPUT_PATH / "health_encoder.pkl")

print("\nPreprocessing objects saved successfully!")

print("✓ scaler.pkl")
print("✓ charging_encoder.pkl")
print("✓ health_encoder.pkl")