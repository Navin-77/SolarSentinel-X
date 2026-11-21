import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
print("REAL SENSOR AUTOENCODER TRAINING (HEALTHY MANIFOLD)")
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
# SELECT HEALTHY DATA (DAYLIGHT + NIGHT/STANDBY)
# ============================================================

df_healthy = df[df["System_Health"] == "Healthy"].copy()

print("\nHealthy samples available for Autoencoder training:")
print(f"Total Healthy rows: {len(df_healthy)}")

df_healthy_features = df_healthy[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
).dropna()

print(f"Valid numeric Healthy rows: {len(df_healthy_features)}")

X_healthy = df_healthy_features.values.astype(np.float32)


# ============================================================
# TRAIN / VALIDATION SPLIT (LEAK-FREE)
# ============================================================

X_train_raw, X_val_raw = train_test_split(
    X_healthy,
    test_size=0.20,
    random_state=42
)

print(f"Healthy Training samples   : {len(X_train_raw)}")
print(f"Healthy Validation samples : {len(X_val_raw)}")


# ============================================================
# NORMALIZATION (FIT ON TRAIN ONLY)
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

print("\nScaling completed (fit on healthy train split only).")


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

input_dim = X_train.shape[1]

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
    name="sensor_reconstruction"
)(decoded)


# ------------------------------------------------------------
# COMPLETE AUTOENCODER
# ------------------------------------------------------------

autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer,
    name="SolarSentinelX_Real_Autoencoder"
)


# ============================================================
# COMPILE
# ============================================================

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()


# ============================================================
# MODEL PATH & CALLBACKS
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "autoencoder_real.keras"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
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
    X_train,
    X_train,
    validation_data=(X_val, X_val),
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
# RECONSTRUCTION ERROR ON NORMAL DATA
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING RECONSTRUCTION ERROR (NORMAL HEALTHY)")
print("=" * 60)

# Evaluate reconstruction error on healthy data
X_healthy_scaled = scaler.transform(X_healthy)
X_healthy_reconstructed = autoencoder.predict(
    X_healthy_scaled,
    verbose=0
)

healthy_reconstruction_errors = np.mean(
    np.square(
        X_healthy_scaled - X_healthy_reconstructed
    ),
    axis=1
)


# ============================================================
# THRESHOLD CALCULATION
# ============================================================

threshold = float(np.percentile(
    healthy_reconstruction_errors,
    95
))

print(
    "\nNormal Reconstruction Error Mean :",
    f"{np.mean(healthy_reconstruction_errors):.6f}"
)

print(
    "Normal Reconstruction Error Std  :",
    f"{np.std(healthy_reconstruction_errors):.6f}"
)

print(
    "Normal Reconstruction Error Max  :",
    f"{np.max(healthy_reconstruction_errors):.6f}"
)

print(
    "Anomaly Threshold (95th %)       :",
    f"{threshold:.6f}"
)


# ============================================================
# SAVE THRESHOLD & FEATURES
# ============================================================

threshold_path = os.path.join(
    OUTPUT_DIR,
    "real_autoencoder_threshold.pkl"
)

joblib.dump(
    threshold,
    threshold_path
)

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
# SAVE RECONSTRUCTION ERRORS & HISTORY
# ============================================================

error_path = os.path.join(
    DATA_OUTPUT_DIR,
    "reconstruction_errors_real.csv"
)

error_df = pd.DataFrame({
    "Reconstruction_Error": healthy_reconstruction_errors
})
error_df["Anomaly"] = error_df["Reconstruction_Error"] > threshold

error_df.to_csv(
    error_path,
    index=False
)

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

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)
print("[OK] Model     :", model_path)
print("[OK] Scaler    :", scaler_path)
print("[OK] Threshold :", threshold_path)
print("[OK] Features  :", feature_path)
print("[OK] Errors    :", error_path)
print("[OK] History   :", history_path)

print("\n" + "=" * 60)
print("REAL SENSOR AUTOENCODER TRAINING COMPLETED")
print("=" * 60)