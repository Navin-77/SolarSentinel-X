import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "real_lstm"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "lstm",
    "real"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("REAL SENSOR LSTM TRAINING")
print("=" * 60)

X_train = np.load(
    os.path.join(
        DATA_DIR,
        "X_train_real_lstm.npy"
    )
)

X_test = np.load(
    os.path.join(
        DATA_DIR,
        "X_test_real_lstm.npy"
    )
)

y_train = np.load(
    os.path.join(
        DATA_DIR,
        "y_train_real_lstm.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_DIR,
        "y_test_real_lstm.npy"
    )
)


print("\nDataset loaded successfully!")

print("Training input :", X_train.shape)
print("Testing input  :", X_test.shape)

print("Training labels:", y_train.shape)
print("Testing labels :", y_test.shape)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = X_train.shape[1]
NUM_FEATURES = X_train.shape[2]
NUM_CLASSES = 3


# ============================================================
# BUILD LSTM
# ============================================================

model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            SEQUENCE_LENGTH,
            NUM_FEATURES
        )
    ),

    Dropout(0.2),

    LSTM(
        32,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("LSTM MODEL")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "lstm_real_model.keras"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=16,
    callbacks=[
        early_stopping,
        checkpoint
    ],
    verbose=1
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"\nTest Loss     : {loss:.4f}"
)

print(
    f"Test Accuracy : {accuracy:.4f}"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(model_path)

print("\nLSTM model saved successfully!")

print(
    "Model:",
    model_path
)

print("\n" + "=" * 60)
print("REAL SENSOR LSTM TRAINING COMPLETED")
print("=" * 60)