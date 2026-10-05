import os
import sys
import subprocess
from pathlib import Path
import pandas as pd


# ============================================================
# SolarSentinel-X
# MASTER REAL-DATA RETRAINING PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET = PROJECT_ROOT / "dataset" / "real_training" / "real_sensor_dataset.csv"

SCRIPTS = [
    (
        "Random Forest",
        PROJECT_ROOT / "models" / "random_forest" / "train_random_forest_real.py"
    ),
    (
        "LSTM Sequence Creation",
        PROJECT_ROOT / "models" / "lstm" / "real" / "create_real_sequences.py"
    ),
    (
        "LSTM Training",
        PROJECT_ROOT / "models" / "lstm" / "real" / "train_lstm_real.py"
    ),
    (
        "LSTM Evaluation",
        PROJECT_ROOT / "models" / "lstm" / "real" / "evaluate_lstm_real.py"
    ),
    (
        "Autoencoder Training",
        PROJECT_ROOT / "models" / "autoencoder" / "real" / "train_autoencoder_real.py"
    ),
    (
        "Autoencoder Evaluation",
        PROJECT_ROOT / "models" / "autoencoder" / "real" / "evaluate_autoencoder_real.py"
    ),
]


# ============================================================
# DISPLAY
# ============================================================

def separator():
    print("=" * 75)


def run_script(name, script):
    separator()
    print(f"RUNNING: {name}")
    print(f"SCRIPT : {script}")
    separator()

    if not script.exists():
        print(f"\nERROR: Script not found:")
        print(script)
        return False

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT)
    )

    if result.returncode != 0:
        print(f"\n❌ {name} FAILED")
        print(f"Return code: {result.returncode}")
        return False

    print(f"\n✅ {name} COMPLETED")
    return True


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset():
    separator()
    print("STEP 1 — VALIDATING REAL SENSOR DATASET")
    separator()

    if not DATASET.exists():
        print("\n❌ Dataset not found:")
        print(DATASET)
        return False

    try:
        df = pd.read_csv(DATASET)
    except Exception as e:
        print("\n❌ Could not read dataset.")
        print(e)
        return False

    required_features = [
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
        "System_Health",
    ]

    missing = [col for col in required_features if col not in df.columns]

    if missing:
        print("\n❌ Missing required columns:")
        for col in missing:
            print("   -", col)
        return False

    print(f"\nDataset:")
    print(f"  Path       : {DATASET}")
    print(f"  Rows       : {len(df)}")
    print(f"  Columns    : {len(df.columns)}")

    print("\nClass distribution:")
    print(df["System_Health"].value_counts().to_dict())

    required_classes = {"Healthy", "Warning", "Critical"}
    actual_classes = set(df["System_Health"].dropna().unique())

    if not required_classes.issubset(actual_classes):
        print("\n❌ Required health classes are missing.")
        print("Required:", required_classes)
        print("Found   :", actual_classes)
        return False

    numeric_columns = required_features[:-1]

    numeric_ok = True

    for column in numeric_columns:
        converted = pd.to_numeric(df[column], errors="coerce")

        if converted.isna().any():
            count = converted.isna().sum()
            print(
                f"\n❌ Non-numeric/invalid values found in "
                f"{column}: {count}"
            )
            numeric_ok = False

    if not numeric_ok:
        return False

    if df["System_Health"].isna().any():
        print("\n❌ Missing System_Health labels found.")
        return False

    print("\n✅ Dataset validation successful.")
    return True


# ============================================================
# MAIN
# ============================================================

def main():

    separator()
    print("SOLARSENTINEL-X")
    print("MASTER REAL-DATA RETRAINING PIPELINE")
    separator()

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nDataset source:")
    print(DATASET)

    # --------------------------------------------------------
    # Validate dataset before touching any model
    # --------------------------------------------------------

    if not validate_dataset():
        print("\n❌ PIPELINE STOPPED.")
        print("Fix the dataset before retraining.")
        sys.exit(1)

    # --------------------------------------------------------
    # Run training pipeline
    # --------------------------------------------------------

    completed = []

    for name, script in SCRIPTS:

        success = run_script(name, script)

        if not success:
            print("\n" + "=" * 75)
            print("❌ RETRAINING PIPELINE STOPPED")
            print("=" * 75)

            print("\nCompleted before failure:")

            for item in completed:
                print("  ✅", item)

            print("\nFailed:")
            print("  ❌", name)

            sys.exit(1)

        completed.append(name)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    separator()
    print("🎉 SOLARSENTINEL-X RETRAINING COMPLETE")
    separator()

    print("\nCompleted stages:")

    for item in completed:
        print("  ✅", item)

    print("\nDataset used:")
    print(DATASET)

    print("\nThe trained models are now updated using")
    print("the current real sensor CSV.")

    print("\nNext step:")
    print("Run the Streamlit dashboard and perform live validation.")

    separator()


if __name__ == "__main__":
    main()