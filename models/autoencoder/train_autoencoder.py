# ==========================================================
# Imports
# ==========================================================

from pathlib import Path

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import build_autoencoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset" / "processed"

MODEL_DIR = PROJECT_ROOT / "models" / "autoencoder" / "saved_models"
HISTORY_DIR = PROJECT_ROOT / "models" / "autoencoder" / "history"

RESULTS_DIR = PROJECT_ROOT / "results" / "autoencoder"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load Preprocessed Dataset
# ==========================================================

# ==========================================================
# Load Training Dataset
# ==========================================================

X_TRAIN_PATH = DATASET_DIR / "X_train.csv"

if not X_TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Training dataset not found:\n{X_TRAIN_PATH}"
    )

X_train = pd.read_csv(X_TRAIN_PATH)

print("=" * 60)
print("Training Dataset Loaded Successfully")
print("=" * 60)

print(f"Shape : {X_train.shape}")

print("\nFirst 5 Rows:")
print(X_train.head())

# ==========================================================
# Load Labels
# ==========================================================

Y_TRAIN_PATH = DATASET_DIR / "y_train.csv"

if not Y_TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Training labels not found:\n{Y_TRAIN_PATH}"
    )

y_train = pd.read_csv(Y_TRAIN_PATH)

print("\nTraining Labels Loaded Successfully")
print(f"Shape : {y_train.shape}")

# ==========================================================
# Select Healthy Samples
# ==========================================================

HEALTHY_CLASS = 1

healthy_mask = y_train["System_Health"] == HEALTHY_CLASS

X_train_healthy = X_train.loc[healthy_mask].reset_index(drop=True)

print("\n" + "=" * 60)
print("Healthy Samples Selected")
print("=" * 60)

print(f"Total Training Samples   : {len(X_train)}")
print(f"Healthy Training Samples : {len(X_train_healthy)}")
print(f"Feature Dimension        : {X_train_healthy.shape[1]}")

print("\nSystem Health Label Distribution:")
print(y_train["System_Health"].value_counts().sort_index())

# ==========================================================
# Train-Validation Split
# ==========================================================

X_train_autoencoder, X_val_autoencoder = train_test_split(
    X_train_healthy,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print("\n" + "=" * 60)
print("Autoencoder Train-Validation Split")
print("=" * 60)

print(f"Training Samples   : {len(X_train_autoencoder)}")
print(f"Validation Samples : {len(X_val_autoencoder)}")

# ==========================================================
# Build Autoencoder Model
# ==========================================================

input_dimension = X_train_autoencoder.shape[1]

autoencoder = build_autoencoder(input_dimension)

print("\n" + "=" * 60)
print("Autoencoder Model Summary")
print("=" * 60)

autoencoder.summary()

# ==========================================================
# Training Callbacks
# ==========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath=MODEL_DIR / "autoencoder_model.keras",
    monitor="val_loss",
    save_best_only=True
)

# ==========================================================
# Train Autoencoder
# ==========================================================

history = autoencoder.fit(
    X_train_autoencoder,
    X_train_autoencoder,
    validation_data=(
        X_val_autoencoder,
        X_val_autoencoder
    ),
    epochs=100,
    batch_size=32,
    shuffle=True,
    callbacks=[
        early_stopping,
        model_checkpoint
    ],
    verbose=1
)

# ==========================================================
# Save Training History
# ==========================================================

history_df = pd.DataFrame(history.history)

history_path = HISTORY_DIR / "training_history.csv"

history_df.to_csv(history_path, index=False)

print("\n" + "=" * 60)
print("Training Completed Successfully")
print("=" * 60)

print(f"Training history saved to:\n{history_path}")

# ==========================================================
# Plot Training History
# ==========================================================

#plt.figure(figsize=(10, 6))

#plt.plot(
#    history.history["loss"],
#    label="Training Loss",
#    linewidth=2
#)

#plt.plot(
#    history.history["val_loss"],
#    label="Validation Loss",
#    linewidth=2
#)

#plt.title("Autoencoder Training Loss")

#plt.xlabel("Epoch")

#plt.ylabel("Loss")

#plt.legend()

#plt.grid(True)

#loss_plot_path = RESULTS_DIR / "loss_curve.png"

#plt.savefig(
#    loss_plot_path,
#    dpi=300,
#    bbox_inches="tight"
#)

#plt.close()

#print("\nLoss curve saved successfully!")

#print(loss_plot_path)

# ==========================================================
# Reconstruction Error on Healthy Validation Data
# ==========================================================

reconstructed = autoencoder.predict(
    X_val_autoencoder,
    verbose=0
)

reconstruction_error = np.mean(
    np.square(
        X_val_autoencoder.values - reconstructed
    ),
    axis=1
)

print("\n" + "=" * 60)
print("Reconstruction Error Statistics")
print("=" * 60)

print(f"Minimum Error : {reconstruction_error.min():.6f}")
print(f"Maximum Error : {reconstruction_error.max():.6f}")
print(f"Average Error : {reconstruction_error.mean():.6f}")
print(f"Std Deviation : {reconstruction_error.std():.6f}")

# ==========================================================
# Calculate Anomaly Threshold
# ==========================================================

threshold = reconstruction_error.mean() + (3 * reconstruction_error.std())

print("\n" + "=" * 60)
print("Anomaly Threshold")
print("=" * 60)

print(f"Threshold : {threshold:.6f}")

# ==========================================================
# Save Threshold
# ==========================================================

threshold_path = MODEL_DIR / "threshold.npy"

np.save(threshold_path, threshold)

print("\nThreshold saved successfully!")

print(threshold_path)

# ==========================================================
# Load Test Dataset
# ==========================================================

X_TEST_PATH = DATASET_DIR / "X_test.csv"
Y_TEST_PATH = DATASET_DIR / "y_test.csv"

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH)

print("\n" + "=" * 60)
print("Test Dataset Loaded Successfully")
print("=" * 60)

print(f"X_test Shape : {X_test.shape}")
print(f"y_test Shape : {y_test.shape}")

# ==========================================================
# Reconstruction Error on Test Dataset
# ==========================================================

test_reconstructed = autoencoder.predict(
    X_test,
    verbose=0
)

test_reconstruction_error = np.mean(
    np.square(
        X_test.values - test_reconstructed
    ),
    axis=1
)

print("\n" + "=" * 60)
print("Test Reconstruction Error Statistics")
print("=" * 60)

print(f"Minimum Error : {test_reconstruction_error.min():.6f}")
print(f"Maximum Error : {test_reconstruction_error.max():.6f}")
print(f"Average Error : {test_reconstruction_error.mean():.6f}")
print(f"Std Deviation : {test_reconstruction_error.std():.6f}")

# ==========================================================
# Detect Anomalies
# ==========================================================

predicted_anomaly = (
    test_reconstruction_error > threshold
).astype(int)

print("\n" + "=" * 60)
print("Anomaly Detection Summary")
print("=" * 60)

normal_count = np.sum(predicted_anomaly == 0)
anomaly_count = np.sum(predicted_anomaly == 1)

print(f"Normal Samples  : {normal_count}")
print(f"Anomaly Samples : {anomaly_count}")
print(f"Total Samples   : {len(predicted_anomaly)}")

# ==========================================================
# Evaluate Anomaly Detection
# ==========================================================

# Healthy = 1
# Warning (0) and Critical (2) are considered anomalies

true_anomaly = (
    y_test["System_Health"] != 1
).astype(int)

accuracy = accuracy_score(
    true_anomaly,
    predicted_anomaly
)

precision = precision_score(
    true_anomaly,
    predicted_anomaly,
    zero_division=0
)

recall = recall_score(
    true_anomaly,
    predicted_anomaly,
    zero_division=0
)

f1 = f1_score(
    true_anomaly,
    predicted_anomaly,
    zero_division=0
)

cm = confusion_matrix(
    true_anomaly,
    predicted_anomaly
)

print("\n" + "=" * 60)
print("Autoencoder Evaluation")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

# ==========================================================
# Save Prediction Results
# ==========================================================

results_df = X_test.copy()

results_df["Actual_Label"] = true_anomaly.values
results_df["Predicted_Label"] = predicted_anomaly
results_df["Reconstruction_Error"] = test_reconstruction_error

results_path = RESULTS_DIR / "autoencoder_predictions.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\nPrediction results saved successfully!")
print(results_path)