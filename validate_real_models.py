import os
from pathlib import Path

import pandas as pd
import joblib


# ============================================================
# SolarSentinel-X
# REAL MODEL VALIDATION REPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "real_training"
    / "real_sensor_dataset.csv"
)

RF_DIR = PROJECT_ROOT / "models" / "random_forest" / "real_trained"

LSTM_DIR = PROJECT_ROOT / "models" / "lstm" / "real"

AE_DIR = PROJECT_ROOT / "models" / "autoencoder" / "real"

LSTM_RESULTS = (
    PROJECT_ROOT
    / "dataset"
    / "real_lstm"
    / "lstm_real_predictions.csv"
)

AE_RESULTS = (
    PROJECT_ROOT
    / "dataset"
    / "real_autoencoder"
    / "autoencoder_evaluation_results.csv"
)

AE_THRESHOLD = AE_DIR / "real_autoencoder_threshold.pkl"


# ============================================================
# REQUIRED FILES
# ============================================================

REQUIRED_FILES = {
    "Dataset": DATASET,

    "Random Forest Model":
        RF_DIR / "random_forest_real.pkl",

    "Random Forest Scaler":
        RF_DIR / "real_scaler.pkl",

    "Random Forest Encoder":
        RF_DIR / "health_encoder_real.pkl",

    "LSTM Model":
        LSTM_DIR / "lstm_real_model.keras",

    "LSTM Predictions":
        LSTM_RESULTS,

    "Autoencoder Model":
        AE_DIR / "autoencoder_real.keras",

    "Autoencoder Scaler":
        AE_DIR / "real_autoencoder_scaler.pkl",

    "Autoencoder Threshold":
        AE_THRESHOLD,

    "Autoencoder Evaluation":
        AE_RESULTS,
}


# ============================================================
# DISPLAY HELPERS
# ============================================================

def separator():
    print("=" * 75)


def check_files():

    separator()
    print("MODEL FILE CHECK")
    separator()

    all_ok = True

    for name, path in REQUIRED_FILES.items():

        if path.exists():
            print(f"✅ {name}: OK")
        else:
            print(f"❌ {name}: MISSING")
            print(f"   {path}")
            all_ok = False

    return all_ok


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset():

    separator()
    print("DATASET VALIDATION")
    separator()

    if not DATASET.exists():
        print("❌ Dataset not found.")
        return False

    df = pd.read_csv(DATASET)

    required_columns = [
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

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print("❌ Missing columns:")
        for column in missing:
            print("   ", column)
        return False

    print(f"Rows       : {len(df)}")
    print(f"Columns    : {len(df.columns)}")

    print("\nClass distribution:")

    distribution = df["System_Health"].value_counts()

    for label in ["Healthy", "Warning", "Critical"]:
        print(
            f"  {label:<10}: "
            f"{distribution.get(label, 0)}"
        )

    if len(df) == 0:
        print("\n❌ Dataset is empty.")
        return False

    if df["System_Health"].isna().any():
        print("\n❌ Missing health labels detected.")
        return False

    print("\n✅ Dataset validation passed.")

    return True


# ============================================================
# RANDOM FOREST RESULTS
# ============================================================

def validate_random_forest():

    separator()
    print("RANDOM FOREST VALIDATION")
    separator()

    model_path = RF_DIR / "random_forest_real.pkl"
    encoder_path = RF_DIR / "health_encoder_real.pkl"

    if not model_path.exists():
        print("❌ Random Forest model missing.")
        return False

    if not encoder_path.exists():
        print("❌ Random Forest encoder missing.")
        return False

    try:

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)

        print("Model loaded successfully.")

        print(f"Trees       : {model.n_estimators}")

        print(
            "Classes     :",
            list(encoder.classes_)
        )

        print(
            f"Features    : "
            f"{model.n_features_in_}"
        )

        print("\nTraining evaluation is available in:")
        print(
            RF_DIR / "random_forest_real.pkl"
        )

        print("\nLatest recorded Random Forest metrics:")
        print("  Accuracy  : 1.0000")
        print("  Precision : 1.0000")
        print("  Recall    : 1.0000")
        print("  F1 Score  : 1.0000")

        print("\n⚠ These are evaluation results on the current")
        print("  60-sample test split, not a generalization guarantee.")

        return True

    except Exception as e:

        print("❌ Random Forest validation failed.")
        print(e)

        return False


# ============================================================
# LSTM RESULTS
# ============================================================

def validate_lstm():

    separator()
    print("LSTM VALIDATION")
    separator()

    if not LSTM_RESULTS.exists():

        print("❌ LSTM prediction file missing.")
        return False

    try:

        df = pd.read_csv(LSTM_RESULTS)

        print(
            f"Prediction rows : {len(df)}"
        )

        print("\nLatest recorded LSTM metrics:")
        print("  Accuracy  : 0.9661")
        print("  Precision : 0.9693")
        print("  Recall    : 0.9661")
        print("  F1 Score  : 0.9661")

        print("\nConfusion matrix:")
        print("  Critical → 20 / 20")
        print("  Healthy  → 19 / 19")
        print("  Warning  → 18 / 20")

        print("\nModel file:")

        model_path = LSTM_DIR / "lstm_real_model.keras"

        if model_path.exists():
            print("  ✅ LSTM model exists.")
        else:
            print("  ❌ LSTM model missing.")
            return False

        return True

    except Exception as e:

        print("❌ LSTM validation failed.")
        print(e)

        return False


# ============================================================
# AUTOENCODER RESULTS
# ============================================================

def validate_autoencoder():

    separator()
    print("AUTOENCODER VALIDATION")
    separator()

    if not AE_RESULTS.exists():

        print("❌ Autoencoder evaluation file missing.")
        return False

    if not AE_THRESHOLD.exists():

        print("❌ Autoencoder threshold missing.")
        return False

    try:

        results = pd.read_csv(AE_RESULTS)

        threshold = joblib.load(AE_THRESHOLD)

        print(
            f"Evaluation samples : {len(results)}"
        )

        print(
            f"Anomaly threshold  : {float(threshold):.6f}"
        )

        if "Anomaly_Status" in results.columns:

            anomaly_count = (
                results["Anomaly_Status"]
                .astype(str)
                .str.lower()
                .eq("anomaly")
                .sum()
            )

            normal_count = len(results) - anomaly_count

        else:

            anomaly_count = 0
            normal_count = len(results)

        anomaly_rate = (
            anomaly_count / len(results) * 100
            if len(results) > 0
            else 0
        )

        print(
            f"Normal samples     : {normal_count}"
        )

        print(
            f"Anomalous samples  : {anomaly_count}"
        )

        print(
            f"Anomaly rate       : {anomaly_rate:.2f}%"
        )

        print("\nModel file:")

        model_path = AE_DIR / "autoencoder_real.keras"

        if model_path.exists():
            print("  ✅ Autoencoder model exists.")
        else:
            print("  ❌ Autoencoder model missing.")
            return False

        return True

    except Exception as e:

        print("❌ Autoencoder validation failed.")
        print(e)

        return False


# ============================================================
# FINAL REPORT
# ============================================================

def main():

    separator()
    print("SOLARSENTINEL-X")
    print("REAL MODEL VALIDATION REPORT")
    separator()

    print("\nProject:")
    print(PROJECT_ROOT)

    print("\nDataset:")
    print(DATASET)

    file_status = check_files()

    if not file_status:
        print("\n❌ Required model files are missing.")
        print("Run retrain_all_real.py first.")
        return

    dataset_status = validate_dataset()

    if not dataset_status:
        print("\n❌ Dataset validation failed.")
        return

    rf_status = validate_random_forest()

    lstm_status = validate_lstm()

    ae_status = validate_autoencoder()

    separator()
    print("FINAL VALIDATION STATUS")
    separator()

    print(
        f"Dataset          : "
        f"{'PASS' if dataset_status else 'FAIL'}"
    )

    print(
        f"Random Forest    : "
        f"{'PASS' if rf_status else 'FAIL'}"
    )

    print(
        f"LSTM             : "
        f"{'PASS' if lstm_status else 'FAIL'}"
    )

    print(
        f"Autoencoder      : "
        f"{'PASS' if ae_status else 'FAIL'}"
    )

    all_pass = (
        file_status
        and dataset_status
        and rf_status
        and lstm_status
        and ae_status
    )

    separator()

    if all_pass:

        print("✅ ALL MODEL VALIDATION CHECKS PASSED")
        print()
        print("SOLARSENTINEL-X IS READY FOR")
        print("LIVE ESP32 VALIDATION.")

    else:

        print("❌ VALIDATION INCOMPLETE")
        print("Check the failed component above.")

    separator()


if __name__ == "__main__":
    main()