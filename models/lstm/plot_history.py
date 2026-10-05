from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

HISTORY_PATH = (
    BASE_DIR
    / "models"
    / "lstm"
    / "history"
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
# Load Training History
# ==========================================================

def load_history():

    print("=" * 60)
    print("Loading Training History...")
    print("=" * 60)

    history = pd.read_csv(
        HISTORY_PATH / "training_history.csv"
    )

    print("✓ History Loaded\n")

    return history


# ==========================================================
# Plot Loss Curve
# ==========================================================

def plot_loss(history):

    plt.figure(figsize=(8,5))

    plt.plot(
        history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.title("LSTM Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        RESULT_PATH / "loss_curve.png"
    )

    plt.close()

    print("✓ Loss Curve Saved")


# ==========================================================
# Plot MAE Curve
# ==========================================================

def plot_mae(history):

    plt.figure(figsize=(8,5))

    plt.plot(
        history["mae"],
        label="Training MAE"
    )

    plt.plot(
        history["val_mae"],
        label="Validation MAE"
    )

    plt.title("LSTM Mean Absolute Error")

    plt.xlabel("Epoch")

    plt.ylabel("MAE")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        RESULT_PATH / "mae_curve.png"
    )

    plt.close()

    print("✓ MAE Curve Saved")


# ==========================================================
# Main
# ==========================================================

def main():

    history = load_history()

    plot_loss(history)

    plot_mae(history)

    print("\nTraining Curves Generated Successfully")


if __name__ == "__main__":

    main()