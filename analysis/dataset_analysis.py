# ==========================================================
# SolarSentinel-X
# Dataset Analysis
# Step 2.1 - Load Dataset and Initial Inspection
# ==========================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Project Root Directory
# ----------------------------------------------------------
# This finds the main SolarSentinel-X project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------
# Dataset Path
# ----------------------------------------------------------
DATASET_PATH = BASE_DIR / "dataset" / "synthetic" / "solar_microgrid_dataset.csv"

# Results folder
results_path = BASE_DIR / "results"
results_path.mkdir(exist_ok=True)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
try:
    df = pd.read_csv(DATASET_PATH)
    print("=" * 70)
    print("✅ DATASET LOADED SUCCESSFULLY")
    print("=" * 70)
except FileNotFoundError:
    print("=" * 70)
    print("❌ ERROR: Dataset file not found.")
    print(f"Expected Location:\n{DATASET_PATH}")
    print("=" * 70)
    exit()

# ----------------------------------------------------------
# Dataset Shape
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("1. DATASET SHAPE")
print("=" * 70)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ----------------------------------------------------------
# Preview First Five Rows
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("2. FIRST FIVE ROWS OF THE DATASET")
print("=" * 70)
print(df.head())

# ----------------------------------------------------------
# Column Names
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("3. COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")
    
# ==========================================================
# Step 2.2 - Missing Value Analysis
# ==========================================================

print("\n" + "=" * 70)
print("4. MISSING VALUE ANALYSIS")
print("=" * 70)

# Count missing values
missing_values = df.isnull().sum()

# Calculate missing percentage
missing_percentage = (missing_values / len(df)) * 100

# Create a summary table
missing_summary = pd.DataFrame({
    "Missing Values": missing_values,
    "Percentage (%)": missing_percentage.round(2)
})

print(missing_summary)

# Display only columns with missing values
print("\n" + "=" * 70)
print("COLUMNS CONTAINING MISSING VALUES")
print("=" * 70)

missing_columns = missing_summary[missing_summary["Missing Values"] > 0]

if missing_columns.empty:
    print("✅ No missing values found in the dataset.")
else:
    print(missing_columns)
    
    
# ==========================================================
# Step 2.3 - Duplicate Row Analysis
# ==========================================================

print("\n" + "=" * 70)
print("5. DUPLICATE ROW ANALYSIS")
print("=" * 70)

# Count duplicate rows
duplicate_count = df.duplicated().sum()

print(f"Total Duplicate Rows : {duplicate_count}")

# Display duplicate rows if any exist
if duplicate_count > 0:
    print("\nDuplicate Records:")
    print(df[df.duplicated()].head())
else:
    print("✅ No duplicate rows found in the dataset.")
    
# ==========================================================
# Step 2.4 - Dataset Information
# ==========================================================

print("\n" + "=" * 70)
print("6. DATASET INFORMATION")
print("=" * 70)

df.info()

# ==========================================================
# Step 2.5 - Statistical Summary
# ==========================================================

print("\n" + "=" * 70)
print("7. STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe().round(2))

# ==========================================================
# Step 2.6 - Feature Range Analysis
# ==========================================================

print("\n" + "=" * 70)
print("8. FEATURE RANGE ANALYSIS")
print("=" * 70)

feature_ranges = {
    "Hour": (0, 23),
    "Day": (1, 31),
    "Solar_Irradiance": (0, 1100),
    "Ambient_Temperature": (20, 45),
    "Humidity": (0, 100),
    "Panel_Voltage": (0, 50),
    "Panel_Current": (0, 15),
    "Panel_Power": (0, 500),
    "Panel_Temperature": (20, 80),
    "MPPT_Input_Voltage": (0, 50),
    "MPPT_Output_Voltage": (0, 50),
    "MPPT_Efficiency": (0.85, 1.00),
    "Battery_Voltage": (10, 15),
    "Battery_Current": (-20, 20),
    "Battery_SOC": (0, 100),
    "Battery_SOH": (70, 100),
    "Battery_Temperature": (15, 60),
    "Load_Power": (0, 200),
    "Load_Current": (0, 20),
    "Power_Balance": (-200, 200),
    "Battery_Stress_Index": (0, 100),
    "Efficiency_Loss": (0, 15)
}

print(f"{'Feature':<25}{'Expected Range':<20}{'Actual Range':<25}{'Status'}")
print("-" * 90)

for feature, expected in feature_ranges.items():
    actual_min = df[feature].min()
    actual_max = df[feature].max()

    status = (
        "✅ PASS"
        if expected[0] <= actual_min and actual_max <= expected[1]
        else "❌ CHECK"
    )

    actual_range = f"({actual_min:.2f}, {actual_max:.2f})"

    print(f"{feature:<25}{str(expected):<20}{actual_range:<25}{status}")
    
# ==========================================================
# Step 2.8 - Fault Type Distribution
# ==========================================================

print("\n" + "=" * 70)
print("10. FAULT TYPE DISTRIBUTION")
print("=" * 70)

# Replace NaN only for analysis
fault_data = df["Fault_Type"].fillna("No Fault")

# Count fault types
fault_counts = fault_data.value_counts()

# Calculate percentages
fault_percent = (fault_counts / len(df) * 100).round(2)

# Summary table
fault_summary = fault_counts.to_frame(name="Count")
fault_summary["Percentage (%)"] = fault_percent

print(fault_summary)

# Plot
plt.figure(figsize=(10, 6))
fault_counts.plot(kind="bar")

plt.title("Fault Type Distribution")
plt.xlabel("Fault Type")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()

# Save graph
plt.savefig(results_path / "fault_type_distribution.png", dpi=300)

plt.show()

print("\n✅ Chart saved to:")
print(results_path / "fault_type_distribution.png")

# ======================================================
# 11. CORRELATION MATRIX
# ======================================================

print("\n" + "=" * 70)
print("11. CORRELATION MATRIX")
print("=" * 70)

# Select only numerical columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

# Print correlation matrix
print(correlation_matrix)

# Plot heatmap
plt.figure(figsize=(16, 12))

plt.imshow(correlation_matrix,
           cmap="coolwarm",
           interpolation="nearest",
           aspect="auto")

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    results_path / "correlation_matrix.png",
    dpi=300
)

plt.show()

print("\n✅ Correlation matrix saved to:")
print(results_path / "correlation_matrix.png")