import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler


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
    "models",
    "autoencoder",
    "real"
)

DATA_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "real_autoencoder"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)


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


# ============================================================
# CONFIGURATION
# ============================================================

LATENT_DIM = 4
EPOCHS = 100
BATCH_SIZE = 16


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("REAL SENSOR AUTOENCODER TRAINING")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df[FEATURES].copy()

df = df.apply(
    pd.to_numeric,
    errors="coerce"
)

df = df.dropna()

print("\nValid numeric rows :", len(df))


# ============================================================
# FEATURE MATRIX
# ============================================================

X = df.values.astype(np.float32)

print(
    "Feature matrix shape :",
    X.shape
)


# ============================================================
# NORMALIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaling completed.")


# ============================================================
# SAVE SCALER
# ============================================================

scaler_path = os.path.join(
    OUTPUT_DIR,
    "real_autoencoder_scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_path
)


# ============================================================
# AUTOENCODER ARCHITECTURE
# ============================================================

input_dim = X_scaled.shape[1]

print("\n" + "=" * 60)
print("AUTOENCODER ARCHITECTURE")
print("=" * 60)

print("Input features :", input_dim)
print("Latent dimension:", LATENT_DIM)


# ------------------------------------------------------------
# ENCODER
# ------------------------------------------------------------

input_layer = Input(
    shape=(input_dim,),
    name="sensor_input"
)

encoded = Dense(
    16,
    activation="relu",
    name="encoder_dense_1"
)(input_layer)

encoded = Dense(
    8,
    activation="relu",
    name="encoder_dense_2"
)(encoded)

latent = Dense(
    LATENT_DIM,
    activation="relu",
    name="latent_space"
)(encoded)


# ------------------------------------------------------------
# DECODER
# ------------------------------------------------------------

decoded = Dense(
    8,
    activation="relu",
    name="decoder_dense_1"
)(latent)

decoded = Dense(
    16,
    activation="relu",
    name="decoder_dense_2"
)(decoded)

output_layer = Dense(
    input_dim,
    activation="linear",
    name="reconstruction"
)(decoded)


# ============================================================
# CREATE MODEL
# ============================================================

autoencoder = Model(
    input_layer,
    output_layer,
    name="SolarSentinelX_Real_Autoencoder"
)


# ============================================================
# COMPILE
# ============================================================

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

autoencoder.summary()


# ============================================================
# MODEL PATH
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "autoencoder_real.keras"
)


# ============================================================
# CALLBACKS
# ============================================================

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
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING AUTOENCODER TRAINING")
print("=" * 60)

history = autoencoder.fit(
    X_scaled,
    X_scaled,
    validation_split=0.20,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[
        early_stopping,
        checkpoint
    ],
    verbose=1
)


# ============================================================
# RECONSTRUCTION
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING RECONSTRUCTION ERROR")
print("=" * 60)

X_reconstructed = autoencoder.predict(
    X_scaled,
    verbose=0
)


# ============================================================
# RECONSTRUCTION ERROR
# ============================================================

reconstruction_errors = np.mean(
    np.square(
        X_scaled - X_reconstructed
    ),
    axis=1
)


# ============================================================
# THRESHOLD
# ============================================================

threshold = np.percentile(
    reconstruction_errors,
    95
)


print(
    "\nReconstruction Error Mean :",
    f"{np.mean(reconstruction_errors):.6f}"
)

print(
    "Reconstruction Error Std  :",
    f"{np.std(reconstruction_errors):.6f}"
)

print(
    "Reconstruction Error Max  :",
    f"{np.max(reconstruction_errors):.6f}"
)

print(
    "Anomaly Threshold (95%)   :",
    f"{threshold:.6f}"
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_path = os.path.join(
    OUTPUT_DIR,
    "real_autoencoder_threshold.pkl"
)

joblib.dump(
    threshold,
    threshold_path
)


# ============================================================
# SAVE FEATURE NAMES
# ============================================================

feature_path = os.path.join(
    OUTPUT_DIR,
    "real_autoencoder_features.csv"
)

pd.DataFrame({
    "Feature": FEATURES
}).to_csv(
    feature_path,
    index=False
)


# ============================================================
# SAVE RECONSTRUCTION ERRORS
# ============================================================

error_path = os.path.join(
    DATA_OUTPUT_DIR,
    "reconstruction_errors_real.csv"
)

error_df = pd.DataFrame({
    "Reconstruction_Error":
        reconstruction_errors
})

error_df["Anomaly"] = (
    error_df["Reconstruction_Error"]
    > threshold
)

error_df.to_csv(
    error_path,
    index=False
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    DATA_OUTPUT_DIR,
    "autoencoder_training_history.csv"
)

pd.DataFrame(
    history.history
).to_csv(
    history_path,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

anomaly_count = np.sum(
    reconstruction_errors > threshold
)

normal_count = len(
    reconstruction_errors
) - anomaly_count


print("\n" + "=" * 60)
print("AUTOENCODER RESULTS")
print("=" * 60)

print(
    "Total samples    :",
    len(reconstruction_errors)
)

print(
    "Normal samples   :",
    normal_count
)

print(
    "Anomalous samples:",
    anomaly_count
)

print(
    "Anomaly rate     :",
    f"{(anomaly_count / len(reconstruction_errors)) * 100:.2f}%"
)


# ============================================================
# FILES
# ============================================================

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    "Model     :",
    model_path
)

print(
    "Scaler    :",
    scaler_path
)

print(
    "Threshold :",
    threshold_path
)

print(
    "Features  :",
    feature_path
)

print(
    "Errors    :",
    error_path
)

print(
    "History   :",
    history_path
)

print("\n" + "=" * 60)
print("REAL SENSOR AUTOENCODER TRAINING COMPLETED")
print("=" * 60)