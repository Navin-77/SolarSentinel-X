import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real_training",
    "real_sensor_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder",
    "real",
    "autoencoder_real.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder",
    "real",
    "real_autoencoder_scaler.pkl"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder",
    "real",
    "real_autoencoder_threshold.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder",
    "real",
    "real_autoencoder_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "real_autoencoder"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("REAL SENSOR AUTOENCODER EVALUATION")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ============================================================
# LOAD FEATURES
# ============================================================

feature_names = pd.read_csv(FEATURE_PATH).iloc[:, 0].dropna().tolist()

print("\nFeatures used by Autoencoder:")
for feature in feature_names:
    print(" -", feature)


X = df[feature_names].apply(pd.to_numeric, errors="coerce")

valid_mask = X.notna().all(axis=1)

X = X[valid_mask].copy()
df_valid = df.loc[valid_mask].copy()

print("\nValid samples :", len(X))
print("Feature shape :", X.shape)


# ============================================================
# LOAD MODEL / SCALER / THRESHOLD
# ============================================================

print("\nLoading Autoencoder...")
model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)
threshold = joblib.load(THRESHOLD_PATH)

print("Model loaded successfully.")
print("Scaler loaded successfully.")
print("Threshold loaded successfully.")

print("\nAnomaly Threshold :", round(float(threshold), 6))


# ============================================================
# SCALE DATA
# ============================================================

X_scaled = scaler.transform(X)


# ============================================================
# RECONSTRUCTION
# ============================================================

print("\nGenerating reconstructions...")

X_reconstructed = model.predict(
    X_scaled,
    verbose=0
)


# ============================================================
# RECONSTRUCTION ERROR
# ============================================================

reconstruction_errors = np.mean(
    np.square(X_scaled - X_reconstructed),
    axis=1
)

anomaly_status = np.where(
    reconstruction_errors > threshold,
    "Anomaly",
    "Normal"
)


# ============================================================
# RESULT DATAFRAME
# ============================================================

results = df_valid.copy()

results["Reconstruction_Error"] = reconstruction_errors
results["Anomaly_Status"] = anomaly_status


# ============================================================
# SUMMARY
# ============================================================

normal_count = np.sum(anomaly_status == "Normal")
anomaly_count = np.sum(anomaly_status == "Anomaly")

print("\n" + "=" * 60)
print("AUTOENCODER RESULTS")
print("=" * 60)

print("\nTotal samples    :", len(results))
print("Normal samples   :", normal_count)
print("Anomalous samples:", anomaly_count)

print(
    "Anomaly rate     : {:.2f}%".format(
        (anomaly_count / len(results)) * 100
    )
)

print("\nReconstruction Error Statistics:")
print("Mean :", round(float(np.mean(reconstruction_errors)), 6))
print("Std  :", round(float(np.std(reconstruction_errors)), 6))
print("Min  :", round(float(np.min(reconstruction_errors)), 6))
print("Max  :", round(float(np.max(reconstruction_errors)), 6))


# ============================================================
# ANOMALY DISTRIBUTION BY SYSTEM HEALTH
# ============================================================

if "System_Health" in results.columns:

    print("\n" + "=" * 60)
    print("ANOMALY DISTRIBUTION BY EXPERIMENTAL CONDITION")
    print("=" * 60)

    distribution = pd.crosstab(
        results["System_Health"],
        results["Anomaly_Status"]
    )

    print("\n", distribution)

    distribution_path = os.path.join(
        OUTPUT_DIR,
        "anomaly_distribution_by_health.csv"
    )

    distribution.to_csv(distribution_path)

    print(
        "\nSaved:",
        distribution_path
    )


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "autoencoder_evaluation_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nEvaluation results saved:")
print(results_path)


# ============================================================
# TOP ANOMALIES
# ============================================================

top_anomalies = results.sort_values(
    "Reconstruction_Error",
    ascending=False
).head(20)

top_path = os.path.join(
    OUTPUT_DIR,
    "top_20_anomalies.csv"
)

top_anomalies.to_csv(
    top_path,
    index=False
)

print("\nTop 20 anomalies saved:")
print(top_path)


print("\nTop anomaly samples:")
print(
    top_anomalies[
        feature_names
        + ["Reconstruction_Error", "Anomaly_Status"]
    ].to_string(index=False)
)


# ============================================================
# VISUALIZATION 1
# RECONSTRUCTION ERROR
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    reconstruction_errors,
    label="Reconstruction Error"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.xlabel("Sample Index")
plt.ylabel("Reconstruction Error")
plt.title("Real Sensor Autoencoder Reconstruction Error")
plt.legend()
plt.grid(True)

error_plot_path = os.path.join(
    OUTPUT_DIR,
    "reconstruction_error_plot.png"
)

plt.tight_layout()
plt.savefig(error_plot_path, dpi=300)
plt.close()

print("\nSaved:")
print(error_plot_path)


# ============================================================
# VISUALIZATION 2
# ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    reconstruction_errors,
    bins=30
)

plt.axvline(
    threshold,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Samples")
plt.title("Distribution of Reconstruction Errors")
plt.legend()
plt.grid(True)

hist_path = os.path.join(
    OUTPUT_DIR,
    "reconstruction_error_distribution.png"
)

plt.tight_layout()
plt.savefig(hist_path, dpi=300)
plt.close()

print("\nSaved:")
print(hist_path)


# ============================================================
# VISUALIZATION 3
# ANOMALY STATUS
# ============================================================

status_counts = results["Anomaly_Status"].value_counts()

plt.figure(figsize=(8, 6))

status_counts.plot(
    kind="bar"
)

plt.xlabel("Status")
plt.ylabel("Number of Samples")
plt.title("Normal vs Anomalous Samples")

plt.tight_layout()

status_plot_path = os.path.join(
    OUTPUT_DIR,
    "normal_vs_anomaly.png"
)

plt.savefig(status_plot_path, dpi=300)
plt.close()

print("\nSaved:")
print(status_plot_path)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("AUTOENCODER EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")
print("1. autoencoder_evaluation_results.csv")
print("2. top_20_anomalies.csv")
print("3. anomaly_distribution_by_health.csv")
print("4. reconstruction_error_plot.png")
print("5. reconstruction_error_distribution.png")
print("6. normal_vs_anomaly.png")

print("\nNext step:")
print("Run the evaluation script and send me the COMPLETE terminal output.")