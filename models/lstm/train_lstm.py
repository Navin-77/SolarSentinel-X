from pathlib import Path

import numpy as np

import pandas as pd

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

SEQUENCE_PATH = BASE_DIR / "dataset" / "sequences"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "lstm"
    / "saved_models"
)

MODEL_PATH.mkdir(parents=True, exist_ok=True)

# ==========================================================
# History Path
# ==========================================================

HISTORY_PATH = (
    BASE_DIR
    / "models"
    / "lstm"
    / "history"
)

HISTORY_PATH.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# Training Configuration
# ==========================================================

EPOCHS = 50

BATCH_SIZE = 32

# ==========================================================
# Load Sequence Dataset
# ==========================================================

def load_dataset():

    print("=" * 60)
    print("Loading LSTM Dataset...")
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

    print("✓ Dataset Loaded Successfully\n")

    print(f"X Shape      : {X_sequences.shape}")
    print(f"SOC Shape    : {y_soc.shape}")
    print(f"SOH Shape    : {y_soh.shape}")
    print(f"RUL Shape    : {y_rul.shape}\n")

    return (
        X_sequences,
        y_soc,
        y_soh,
        y_rul
    )
    
# ==========================================================
# Create Multi-Output Targets
# ==========================================================

def create_targets(
    y_soc,
    y_soh,
    y_rul
):

    print("=" * 60)
    print("Creating Multi-Output Target Matrix...")
    print("=" * 60)

    y_train = np.column_stack(
        (
            y_soc,
            y_soh,
            y_rul
        )
    )

    print("✓ Target Matrix Created\n")

    print(f"Target Shape : {y_train.shape}\n")

    return y_train

# ==========================================================
# Build LSTM Model
# ==========================================================

def build_model(input_shape):

    print("=" * 60)
    print("Building LSTM Model...")
    print("=" * 60)

    model = Sequential([

        Input(shape=input_shape),

        LSTM(
            units=64,
            return_sequences=True
        ),

        Dropout(0.2),

        LSTM(
            units=32
        ),

        Dropout(0.2),

        Dense(
            units=16,
            activation="relu"
        ),

        Dense(
            units=3,
            activation="linear"
        )

    ])

    print("✓ LSTM Model Built Successfully\n")

    return model

# ==========================================================
# Compile Model
# ==========================================================

def compile_model(model):

    print("=" * 60)
    print("Compiling LSTM Model...")
    print("=" * 60)

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    print("✓ Model Compiled Successfully\n")
    
# ==========================================================
# Create Early Stopping Callback
# ==========================================================

def create_early_stopping():

    print("=" * 60)
    print("Creating EarlyStopping Callback...")
    print("=" * 60)

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

        verbose=1

    )

    print("✓ EarlyStopping Ready\n")

    return early_stopping

# ==========================================================
# Create Model Checkpoint Callback
# ==========================================================

def create_model_checkpoint():

    print("=" * 60)
    print("Creating ModelCheckpoint Callback...")
    print("=" * 60)

    checkpoint = ModelCheckpoint(

        filepath=MODEL_PATH / "lstm_model.keras",

        monitor="val_loss",

        save_best_only=True,

        verbose=1

    )

    print("✓ ModelCheckpoint Ready\n")

    return checkpoint
    
# ==========================================================
# Train LSTM Model
# ==========================================================

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    early_stopping,
    model_checkpoint
):

    print("=" * 60)
    print("Training LSTM Model...")
    print("=" * 60)

    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stopping,
            model_checkpoint
        ],

        verbose=1

    )

    print("\n✓ LSTM Training Completed Successfully\n")

    return history

# ==========================================================
# Verify Saved Model
# ==========================================================

def verify_saved_model():

    print("=" * 60)
    print("Verifying Saved Model...")
    print("=" * 60)

    model_file = MODEL_PATH / "lstm_model.keras"

    if model_file.exists():

        print("✓ Best Model Saved Successfully")
        print(model_file)

    else:

        print("✗ Model Not Found")

    print()
    
# ==========================================================
# Save Training History
# ==========================================================

def save_training_history(history):

    print("=" * 60)
    print("Saving Training History...")
    print("=" * 60)

    history_dataframe = pd.DataFrame(
        history.history
    )

    history_file = (
        HISTORY_PATH
        / "training_history.csv"
    )

    history_dataframe.to_csv(

        history_file,

        index=False

    )

    print("✓ Training History Saved Successfully")
    print(history_file)
    print()
    
# ==========================================================
# Split Training and Validation Data
# ==========================================================

def split_dataset(X, y):

    print("=" * 60)
    print("Creating Training and Validation Sets...")
    print("=" * 60)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    print("✓ Dataset Split Completed\n")

    print(f"Training Samples   : {X_train.shape[0]}")
    print(f"Validation Samples : {X_val.shape[0]}\n")

    return (
        X_train,
        X_val,
        y_train,
        y_val
    )
    
    
# ==========================================================
# Main Function
# ==========================================================

def main():

    (
        X_sequences,
        y_soc,
        y_soh,
        y_rul
    ) = load_dataset()
    
    targets = create_targets(
        y_soc,
        y_soh,
        y_rul
    )

    X_train, X_val, y_train, y_val = split_dataset(
        X_sequences,
        targets
    )

    model = build_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    compile_model(model)
    
    early_stopping = create_early_stopping()
    
    model_checkpoint = create_model_checkpoint()
    
    history = train_model(

        model,

        X_train,

        y_train,

        X_val,

        y_val,

        early_stopping,

        model_checkpoint

    )
    
    verify_saved_model()
    
    save_training_history(history)

    model.summary()

    print("=" * 60)
    print("Dataset Ready for LSTM Training")
    print("=" * 60)

    print(f"Training Input Shape   : {X_train.shape}")
    print(f"Training Output Shape  : {y_train.shape}")

    print(f"Validation Input Shape : {X_val.shape}")
    print(f"Validation Output Shape: {y_val.shape}")


if __name__ == "__main__":
    main()