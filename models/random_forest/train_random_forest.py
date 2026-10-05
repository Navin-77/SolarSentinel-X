# =====================================================
# SolarSentinel-X
# Random Forest Training
# =====================================================

from pathlib import Path

import pandas as pd

import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

#import matplotlib.pyplot as plt

# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "processed"

# -----------------------------------------------------
# Load Processed Dataset
# -----------------------------------------------------

print("=" * 70)
print("RANDOM FOREST TRAINING")
print("=" * 70)

X_train = pd.read_csv(DATASET_PATH / "X_train.csv")
X_test = pd.read_csv(DATASET_PATH / "X_test.csv")

y_train = pd.read_csv(DATASET_PATH / "y_train.csv")
y_test = pd.read_csv(DATASET_PATH / "y_test.csv")

# Convert labels to 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print("\nProcessed Dataset Loaded Successfully!\n")

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

print(f"\nTraining Features : {X_train.shape}")
print(f"Testing Features  : {X_test.shape}")

print(f"\nTraining Labels : {y_train.shape}")
print(f"Testing Labels  : {y_test.shape}")

# =====================================================
# Train Random Forest Model
# =====================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")

print(f"Number of Trees : {rf_model.n_estimators}")

# =====================================================
# Model Prediction
# =====================================================

print("\n" + "=" * 70)
print("MODEL PREDICTION")
print("=" * 70)

y_pred = rf_model.predict(X_test)

print("\nPredictions generated successfully!")
print(f"Number of Predictions : {len(y_pred)}")

# =====================================================
# Model Evaluation
# =====================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# Feature Importance
# =====================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

# Create results folder
RESULTS_PATH = BASE_DIR / "results" / "random_forest"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Get feature importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

# Sort in descending order
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Most Important Features:\n")
print(feature_importance.head(10))

# Save CSV
feature_importance.to_csv(
    RESULTS_PATH / "feature_importance.csv",
    index=False
)

# Plot
#plt.figure(figsize=(10, 7))

#plt.barh(
#    feature_importance["Feature"],
#    feature_importance["Importance"]
#)

#plt.xlabel("Importance Score")
#plt.ylabel("Feature")
#plt.title("Random Forest Feature Importance")

#plt.gca().invert_yaxis()

#plt.tight_layout()

#plt.savefig(
#    RESULTS_PATH / "feature_importance.png",
#    dpi=300
#)

#plt.show()

print("\nFeature importance saved successfully!")

# =====================================================
# Save Trained Model
# =====================================================

print("\n" + "=" * 70)
print("SAVING RANDOM FOREST MODEL")
print("=" * 70)

# Create model directory
MODEL_PATH = BASE_DIR / "models" / "random_forest" / "saved_models"
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Save model
joblib.dump(
    rf_model,
    MODEL_PATH / "random_forest_model.pkl"
)

print("\nRandom Forest model saved successfully!")

print(f"Location : {MODEL_PATH}")
print("File     : random_forest_model.pkl")

# =====================================================
# Verify Saved Model
# =====================================================

print("\n" + "=" * 70)
print("VERIFY SAVED MODEL")
print("=" * 70)

loaded_model = joblib.load(
    MODEL_PATH / "random_forest_model.pkl"
)

loaded_predictions = loaded_model.predict(X_test)

same_predictions = (loaded_predictions == y_pred).all()

print("\nModel loaded successfully!")

print(f"Predictions Match : {same_predictions}")

if same_predictions:
    print("\n✅ Saved model verified successfully!")
else:
    print("\n❌ Verification failed!")