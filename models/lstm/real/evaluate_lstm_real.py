import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lstm",
    "real",
    "lstm_real_model.keras"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("REAL SENSOR LSTM EVALUATION")
print("=" * 60)

X_test = np.load(
    os.path.join(
        DATA_DIR,
        "X_test_real_lstm.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_DIR,
        "y_test_real_lstm.npy"
    )
)

print("\nTest data loaded successfully!")

print("Test input shape :", X_test.shape)
print("Test labels shape:", y_test.shape)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained LSTM model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# LABEL NAMES
# ============================================================

class_names = [
    "Critical",
    "Healthy",
    "Warning"
]


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy  : {accuracy:.4f}"
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


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    "Rows    = Actual"
)

print(
    "Columns = Predicted"
)

print()

print("              Predicted")
print("             C    H    W")
print(
    f"Actual C   {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}"
)
print(
    f"       H   {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}"
)
print(
    f"       W   {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

output_path = os.path.join(
    DATA_DIR,
    "lstm_real_predictions.csv"
)

import pandas as pd

results = pd.DataFrame({
    "Actual": [
        class_names[i]
        for i in y_test
    ],
    "Predicted": [
        class_names[i]
        for i in y_pred
    ],
    "Correct": (
        y_test == y_pred
    )
})

results.to_csv(
    output_path,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\nPredictions saved to:")

print(output_path)

print("\n" + "=" * 60)
print("REAL SENSOR LSTM EVALUATION COMPLETED")
print("=" * 60)