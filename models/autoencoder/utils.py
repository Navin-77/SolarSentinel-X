# ==========================================================
# SolarSentinel-X
# Autoencoder Utility Functions
# ==========================================================

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


# ==========================================================
# Build Autoencoder
# ==========================================================

def build_autoencoder(input_dim):
    """
    Builds and returns a fully-connected autoencoder model.
    """

    inputs = Input(shape=(input_dim,))

    # ---------------- Encoder ----------------

    x = Dense(16, activation="relu")(inputs)
    x = Dense(8, activation="relu")(x)

    # ---------------- Bottleneck ----------------

    bottleneck = Dense(4, activation="relu")(x)

    # ---------------- Decoder ----------------

    x = Dense(8, activation="relu")(bottleneck)
    x = Dense(16, activation="relu")(x)

    outputs = Dense(input_dim, activation="linear")(x)

    autoencoder = Model(inputs, outputs, name="SolarSentinelX_Autoencoder")

    autoencoder.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return autoencoder