"""
SolarSentinel-X
Phase 5 - LSTM

Step 5.2
Sequence Generation

Loads the processed training dataset
and prepares it for LSTM sequence creation.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

TRAIN_DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "X_train.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "dataset"
    / "sequences"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Configuration
# ==========================================================

SEQUENCE_LENGTH = 24


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    print("=" * 60)
    print("Loading Processed Training Dataset...")
    print("=" * 60)

    dataset = pd.read_csv(TRAIN_DATA_PATH)

    print("✓ Dataset Loaded Successfully")
    print(f"Samples          : {dataset.shape[0]}")
    print(f"Features         : {dataset.shape[1]}")
    print(f"Sequence Length  : {SEQUENCE_LENGTH}\n")

    return dataset

# ==========================================================
# Create Sequences
# ==========================================================

def create_sequences(dataset):

    print("=" * 60)
    print("Creating LSTM Sequences...")
    print("=" * 60)

    data = dataset.values

    X_sequences = []

    for i in range(len(data) - SEQUENCE_LENGTH):

        sequence = data[i : i + SEQUENCE_LENGTH]

        X_sequences.append(sequence)

    X_sequences = np.array(X_sequences)

    print("✓ Sequence Generation Completed\n")

    print(f"Number of Sequences : {X_sequences.shape[0]}")
    print(f"Sequence Length     : {X_sequences.shape[1]}")
    print(f"Features            : {X_sequences.shape[2]}\n")

    return X_sequences

# ==========================================================
# Create Battery SOC Targets
# ==========================================================

def create_soc_targets(dataset):

    print("=" * 60)
    print("Creating Battery SOC Targets...")
    print("=" * 60)

    y_soc = []

    for i in range(len(dataset) - SEQUENCE_LENGTH):

        target = dataset.iloc[
            i + SEQUENCE_LENGTH
        ]["Battery_SOC"]

        y_soc.append(target)

    y_soc = np.array(y_soc)

    print("✓ Battery SOC Targets Created\n")

    print(f"Target Shape : {y_soc.shape}\n")

    return y_soc

# ==========================================================
# Create Battery SOH Targets
# ==========================================================

def create_soh_targets(dataset):

    print("=" * 60)
    print("Creating Battery SOH Targets...")
    print("=" * 60)

    y_soh = []

    for i in range(len(dataset) - SEQUENCE_LENGTH):

        target = dataset.iloc[
            i + SEQUENCE_LENGTH
        ]["Battery_SOH"]

        y_soh.append(target)

    y_soh = np.array(y_soh)

    print("✓ Battery SOH Targets Created\n")

    print(f"Target Shape : {y_soh.shape}\n")

    return y_soh

# ==========================================================
# Create Remaining Useful Life (RUL) Targets
# ==========================================================

def create_rul_targets(dataset):

    print("=" * 60)
    print("Creating Remaining Useful Life Targets...")
    print("=" * 60)

    y_rul = []

    for i in range(len(dataset) - SEQUENCE_LENGTH):

        target = dataset.iloc[
            i + SEQUENCE_LENGTH
        ]["Battery_RUL"]

        y_rul.append(target)

    y_rul = np.array(y_rul)

    print("✓ Remaining Useful Life Targets Created\n")

    print(f"Target Shape : {y_rul.shape}\n")

    return y_rul

# ==========================================================
# Save Sequences
# ==========================================================

def save_sequences(
    X_sequences,
    y_soc,
    y_soh,
    y_rul
):

    print("=" * 60)
    print("Saving LSTM Dataset...")
    print("=" * 60)

    np.save(
        OUTPUT_DIR / "X_sequences.npy",
        X_sequences
    )

    np.save(
        OUTPUT_DIR / "y_soc.npy",
        y_soc
    )

    np.save(
        OUTPUT_DIR / "y_soh.npy",
        y_soh
    )

    np.save(
        OUTPUT_DIR / "y_rul.npy",
        y_rul
    )

    print("✓ X_sequences.npy Saved")
    print("✓ y_soc.npy Saved")
    print("✓ y_soh.npy Saved")
    print("✓ y_rul.npy Saved\n")


# ==========================================================
# Main
# ==========================================================

def main():

    dataset = load_dataset()

    X_sequences = create_sequences(dataset)

    y_soc = create_soc_targets(dataset)

    y_soh = create_soh_targets(dataset)

    y_rul = create_rul_targets(dataset)

    print(f"First SOC Target : {y_soc[0]:.4f}")
    print(f"First SOH Target : {y_soh[0]:.4f}")
    print(f"First RUL Target : {y_rul[0]:.4f}")

    save_sequences(
        X_sequences,
        y_soc,
        y_soh,
        y_rul
    )
    
    print("=" * 60)
    print("Sequence Generation Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()