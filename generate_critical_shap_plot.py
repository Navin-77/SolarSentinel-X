import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ============================================================
# SOLARSENTINEL-X
# CRITICAL CLASS SHAP SUMMARY
# Uses already-generated SHAP values
# No SHAP / Numba import required
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SHAP_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real_shap",
    "shap_values_real.npy"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real_training",
    "real_sensor_dataset.csv"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest",
    "real_trained",
    "real_scaler.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest",
    "real_trained",
    "feature_names_real.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real_shap",
    "shap_summary_Critical_current.png"
)

# ------------------------------------------------------------
# Load existing SHAP values
# ------------------------------------------------------------

print("=" * 65)
print("SOLARSENTINEL-X CRITICAL SHAP PLOT")
print("=" * 65)

shap_values = np.load(SHAP_PATH)

print("SHAP array shape:", shap_values.shape)

# ------------------------------------------------------------
# Load dataset and feature names
# ------------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

feature_names = pd.read_csv(
    FEATURE_PATH
).iloc[:, 0].dropna().tolist()

X = df[feature_names].apply(
    pd.to_numeric,
    errors="coerce"
)

valid_mask = X.notna().all(axis=1)

X = X.loc[valid_mask].copy()

# ------------------------------------------------------------
# Determine SHAP array format
# ------------------------------------------------------------

if shap_values.ndim != 3:
    raise ValueError(
        f"Expected 3D SHAP array, got {shap_values.shape}"
    )

# Current expected format:
# (samples, features, classes)

if shap_values.shape[0] == len(X):

    critical_shap = shap_values[:, :, 0]

else:

    # Alternative:
    # (classes, samples, features)

    critical_shap = shap_values[0, :, :]

print("Critical SHAP shape:", critical_shap.shape)

# ------------------------------------------------------------
# Sort features by mean absolute SHAP value
# ------------------------------------------------------------

importance = np.mean(
    np.abs(critical_shap),
    axis=0
)

order = np.argsort(importance)[::-1]

feature_names_sorted = [
    feature_names[i] for i in order
]

shap_sorted = critical_shap[:, order]

# ------------------------------------------------------------
# Normalize feature values for colour representation
# ------------------------------------------------------------

X_values = X.iloc[:, order].values.astype(float)

feature_min = np.nanmin(X_values, axis=0)
feature_max = np.nanmax(X_values, axis=0)

normalized = (
    X_values - feature_min
) / (
    feature_max - feature_min + 1e-12
)

# ------------------------------------------------------------
# Create SHAP-style summary plot
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 7)
)

rng = np.random.default_rng(42)

for row in range(len(feature_names_sorted)):

    values = shap_sorted[:, row]

    # Limit excessive points only for visual clarity
    if len(values) > 300:
        indices = rng.choice(
            len(values),
            300,
            replace=False
        )
    else:
        indices = np.arange(len(values))

    x_values = values[indices]

    # Small vertical jitter
    y_values = (
        row
        + rng.uniform(
            -0.22,
            0.22,
            len(indices)
        )
    )

    colors = normalized[
        indices,
        row
    ]

    scatter = ax.scatter(
        x_values,
        y_values,
        c=colors,
        cmap="coolwarm",
        s=18,
        alpha=0.75,
        edgecolors="none"
    )

# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.axvline(
    0,
    linewidth=1
)

ax.set_yticks(
    range(len(feature_names_sorted))
)

ax.set_yticklabels(
    feature_names_sorted
)

ax.invert_yaxis()

ax.set_xlabel(
    "SHAP value (impact on model output)"
)

ax.set_ylabel(
    "Sensor Feature"
)

ax.set_title(
    "SHAP Summary - Critical Class"
)

# Colorbar
cbar = plt.colorbar(
    scatter,
    ax=ax
)

cbar.set_label(
    "Feature value"
)

cbar.set_ticks(
    [0, 1]
)

cbar.set_ticklabels(
    ["Low", "High"]
)

plt.tight_layout()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("=" * 65)
print("CRITICAL SHAP PLOT GENERATED SUCCESSFULLY")
print("=" * 65)
print()
print("Saved to:")
print(OUTPUT_PATH)
print()
print("Use this image for FIG. 12.")