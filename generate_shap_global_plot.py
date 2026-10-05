import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SOLARSENTINEL-X
# GLOBAL SHAP FEATURE IMPORTANCE PLOT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real_shap",
    "global_shap_feature_importance.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real_shap",
    "shap_global_feature_importance.png"
)

# Load SHAP importance data
df = pd.read_csv(CSV_PATH)

# Sort from lowest to highest for horizontal bar chart
df = df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=True
)

# Create figure
plt.figure(figsize=(10, 6))

plt.barh(
    df["Feature"],
    df["Mean_Absolute_SHAP"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Sensor Feature")
plt.title("Global SHAP Feature Importance")

plt.tight_layout()

# Save high-resolution image
plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 60)
print("GLOBAL SHAP PLOT GENERATED")
print("=" * 60)
print("Saved to:")
print(OUTPUT_PATH)