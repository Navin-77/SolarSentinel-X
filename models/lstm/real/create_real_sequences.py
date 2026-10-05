import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real_training",
    "real_sensor_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "real_lstm"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# REAL SENSOR FEATURES
# ============================================================

FEATURES = [
    "LDR",
    "DHT22_Temperature",
    "DHT22_Humidity",
    "DS18B20_Temperature",
    "Solar_Voltage",
    "Solar_Current",
    "Solar_Power",
    "Battery_Voltage",
    "Battery_Current",
    "Battery_Power"
]

TARGET = "System_Health"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("REAL SENSOR LSTM SEQUENCE CREATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ============================================================
# VALIDATION
# ============================================================

required_columns = FEATURES + [TARGET]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df[required_columns].copy()

df = df.dropna()

df = df[
    df[TARGET].isin(
        ["Healthy", "Warning", "Critical"]
    )
]

print("\nValid rows :", len(df))

print("\nHealth distribution:")
print(df[TARGET].value_counts())


# ============================================================
# LABEL ENCODING
# ============================================================

label_mapping = {
    "Critical": 0,
    "Healthy": 1,
    "Warning": 2
}

df["Health_Label"] = df[TARGET].map(label_mapping)


# ============================================================
# FEATURE MATRIX
# ============================================================

X = df[FEATURES].values.astype(np.float32)
y = df["Health_Label"].values.astype(np.int32)


# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# CREATE SEQUENCES
# ============================================================

SEQUENCE_LENGTH = 10

X_sequences = []
y_sequences = []

for i in range(
    len(X_scaled) - SEQUENCE_LENGTH + 1
):

    sequence = X_scaled[
        i:i + SEQUENCE_LENGTH
    ]

    # Target = health state of final
    # measurement in the sequence
    target = y[
        i + SEQUENCE_LENGTH - 1
    ]

    X_sequences.append(sequence)
    y_sequences.append(target)


X_sequences = np.array(
    X_sequences,
    dtype=np.float32
)

y_sequences = np.array(
    y_sequences,
    dtype=np.int32
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.20,
    random_state=42,
    stratify=y_sequences
)


# ============================================================
# SAVE DATA
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train_real_lstm.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test_real_lstm.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train_real_lstm.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test_real_lstm.npy"
    ),
    y_test
)


joblib.dump(
    scaler,
    os.path.join(
        OUTPUT_DIR,
        "real_lstm_scaler.pkl"
    )
)


pd.DataFrame({
    "Feature": FEATURES
}).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "real_lstm_features.csv"
    ),
    index=False
)


joblib.dump(
    label_mapping,
    os.path.join(
        OUTPUT_DIR,
        "real_lstm_label_mapping.pkl"
    )
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SEQUENCE CREATION COMPLETED")
print("=" * 60)

print("Sequence length :", SEQUENCE_LENGTH)
print("Features        :", len(FEATURES))

print("\nTotal sequences :", len(X_sequences))

print(
    "Training shape  :",
    X_train.shape
)

print(
    "Testing shape   :",
    X_test.shape
)

print(
    "Training labels :",
    y_train.shape
)

print(
    "Testing labels  :",
    y_test.shape
)

print("\nSaved to:")
print(OUTPUT_DIR)

print("=" * 60)