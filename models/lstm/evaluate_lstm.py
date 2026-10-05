from pathlib import Path

import numpy as np

from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pandas as pd

import matplotlib.pyplot as plt

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "lstm"
    / "saved_models"
)

SEQUENCE_PATH = (
    BASE_DIR
    / "dataset"
    / "sequences"
)

RESULT_PATH = (
    BASE_DIR
    / "results"
    / "lstm"
)

RESULT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# Load Saved Model
# ==========================================================

def load_lstm_model():

    print("=" * 60)
    print("Loading Saved LSTM Model...")
    print("=" * 60)

    model = load_model(

        MODEL_PATH / "lstm_model.keras"

    )

    print("✓ Model Loaded Successfully\n")

    return model

# ==========================================================
# Load Evaluation Dataset
# ==========================================================

def load_evaluation_dataset():

    print("=" * 60)
    print("Loading Evaluation Dataset...")
    print("=" * 60)

    X_sequences = np.load(
        SEQUENCE_PATH / "X_sequences.npy"
    )

    y_soc = np.load(
        SEQUENCE_PATH / "y_soc.npy"
    )

    y_soh = np.load(
        SEQUENCE_PATH / "y_soh.npy"
    )

    y_rul = np.load(
        SEQUENCE_PATH / "y_rul.npy"
    )

    targets = np.column_stack(

        (
            y_soc,
            y_soh,
            y_rul
        )

    )

    X_train, X_val, y_train, y_val = train_test_split(

        X_sequences,

        targets,

        test_size=0.20,

        random_state=42,

        shuffle=True

    )

    print("✓ Evaluation Dataset Loaded\n")

    print(f"Validation Input Shape  : {X_val.shape}")
    print(f"Validation Output Shape : {y_val.shape}\n")

    return X_val, y_val

# ==========================================================
# Generate Predictions
# ==========================================================

def generate_predictions(
    model,
    X_val
):

    print("=" * 60)
    print("Generating Predictions...")
    print("=" * 60)

    predictions = model.predict(
        X_val,
        verbose=1
    )

    print("✓ Predictions Generated\n")

    print(f"Prediction Shape : {predictions.shape}\n")

    return predictions

# ==========================================================
# Calculate Evaluation Metrics
# ==========================================================

def calculate_metrics(y_true, predictions):

    print("=" * 60)
    print("Calculating Evaluation Metrics...")
    print("=" * 60)

    target_names = [
        "SOC",
        "SOH",
        "RUL"
    ]

    metrics = []

    for i, target in enumerate(target_names):

        mae = mean_absolute_error(
            y_true[:, i],
            predictions[:, i]
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true[:, i],
                predictions[:, i]
            )
        )

        r2 = r2_score(
            y_true[:, i],
            predictions[:, i]
        )

        metrics.append(
            [
                target,
                mae,
                rmse,
                r2
            ]
        )

        print(f"\n{target}")

        print(f"MAE  : {mae:.4f}")

        print(f"RMSE : {rmse:.4f}")

        print(f"R²   : {r2:.4f}")

    return pd.DataFrame(

        metrics,

        columns=[
            "Target",
            "MAE",
            "RMSE",
            "R2"
        ]

    )
    
# ==========================================================
# Save Evaluation Metrics
# ==========================================================

def save_metrics(metrics_df):

    print("=" * 60)
    print("Saving Evaluation Metrics...")
    print("=" * 60)

    metrics_file = (
        RESULT_PATH /
        "evaluation_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    print("✓ Evaluation Metrics Saved Successfully")
    print(f"Location : {metrics_file}\n")
    
# ==========================================================
# Plot Actual vs Predicted
# ==========================================================

def plot_predictions(y_true, predictions):

    print("=" * 60)
    print("Plotting Actual vs Predicted Graphs...")
    print("=" * 60)

    targets = [
        "SOC",
        "SOH",
        "RUL"
    ]

    filenames = [
        "soc_prediction.png",
        "soh_prediction.png",
        "rul_prediction.png"
    ]

    for i in range(3):

        plt.figure(figsize=(10, 5))

        plt.plot(
            y_true[:, i],
            label="Actual"
        )

        plt.plot(
            predictions[:, i],
            label="Predicted"
        )

        plt.title(f"{targets[i]} : Actual vs Predicted")

        plt.xlabel("Sample")

        plt.ylabel(targets[i])

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            RESULT_PATH / filenames[i]
        )

        plt.close()

        print(f"✓ {filenames[i]} Saved")

    print()

# ==========================================================
# Main
# ==========================================================

def main():

    model = load_lstm_model()

    X_val, y_val = load_evaluation_dataset()

    predictions = generate_predictions(

        model,

        X_val

    )
    
    metrics = calculate_metrics(
        y_val,
        predictions
    )
    
    save_metrics(metrics)
    
    plot_predictions(
        y_val,
        predictions
    )
    
    print("=" * 60)
    print("Sample Predictions")
    print("=" * 60)

    for i in range(5):

        print(f"\nSample {i+1}")

        print(f"Actual    : {y_val[i]}")

        print(f"Predicted : {predictions[i]}")

    model.summary()


if __name__ == "__main__":

    main()