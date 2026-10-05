import pandas as pd
import numpy as np

INPUT = "dataset/real_training/real_sensor_dataset.csv"
OUTPUT = "dataset/real_training/real_sensor_dataset.csv"

df = pd.read_csv(INPUT)

print("Original dataset:")
print(df["System_Health"].value_counts())

# --------------------------------------------------
# Select final balanced dataset
# --------------------------------------------------

healthy = df[df["System_Health"] == "Healthy"].copy()
warning = df[df["System_Health"] == "Warning"].copy()
critical = df[df["System_Health"] == "Critical"].copy()

# Take exactly 100 from each class
healthy = healthy.iloc[:100].copy()
warning = warning.iloc[:100].copy()
critical = critical.iloc[:100].copy()

# --------------------------------------------------
# Normal / Healthy feature correction
# --------------------------------------------------

np.random.seed(42)

# LDR: synthetic Normal range
healthy["LDR"] = np.random.randint(860, 951, size=len(healthy))

# DS18B20: synthetic Normal range
healthy["DS18B20_Temperature"] = np.round(
    np.random.uniform(28.0, 30.0, size=len(healthy)),
    2
)

# --------------------------------------------------
# Combine
# --------------------------------------------------

final_df = pd.concat(
    [healthy, warning, critical],
    ignore_index=True
)

# Shuffle rows
final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
final_df.to_csv(OUTPUT, index=False)

print("\n========================================")
print("FINAL DATASET CREATED")
print("========================================")

print("\nShape:", final_df.shape)

print("\nClass distribution:")
print(final_df["System_Health"].value_counts())

print("\nHealthy LDR range:")
print(
    final_df[final_df["System_Health"] == "Healthy"]["LDR"].min(),
    "to",
    final_df[final_df["System_Health"] == "Healthy"]["LDR"].max()
)

print("\nHealthy DS18B20 range:")
print(
    final_df[final_df["System_Health"] == "Healthy"]["DS18B20_Temperature"].min(),
    "to",
    final_df[final_df["System_Health"] == "Healthy"]["DS18B20_Temperature"].max()
)

print("\nMissing values:")
print(final_df.isna().sum())

print("\n========================================")
print("DONE")
print("========================================")