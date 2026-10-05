"""
Model Loader

Loads all trained AI models and preprocessing artifacts
required by the Digital Twin.
"""

from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

import shap


class ModelLoader:
    """Loads all AI models used by the Digital Twin."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent

    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------

    def load_random_forest(self):
        model_path = (
            self.project_root
            / "models"
            / "random_forest"
            / "saved_models"
            / "random_forest_model.pkl"
        )

        return joblib.load(model_path)

    # --------------------------------------------------
    # LSTM
    # --------------------------------------------------

    def load_lstm(self):
        model_path = (
            self.project_root
            / "models"
            / "lstm"
            / "saved_models"
            / "lstm_model.keras"
        )

        return tf.keras.models.load_model(model_path)

    # --------------------------------------------------
    # Autoencoder
    # --------------------------------------------------

    def load_autoencoder(self):
        model_path = (
            self.project_root
            / "models"
            / "autoencoder"
            / "saved_models"
            / "autoencoder_model.keras"
        )

        return tf.keras.models.load_model(model_path)

    def load_threshold(self):
        threshold_path = (
            self.project_root
            / "models"
            / "autoencoder"
            / "saved_models"
            / "threshold.npy"
        )

        return float(np.load(threshold_path))

    # --------------------------------------------------
    # Preprocessing Objects
    # --------------------------------------------------

    def load_scaler(self):
        scaler_path = (
            self.project_root
            / "dataset"
            / "processed"
            / "scaler.pkl"
        )

        return joblib.load(scaler_path)

    def load_health_encoder(self):
        encoder_path = (
            self.project_root
            / "dataset"
            / "processed"
            / "health_encoder.pkl"
        )

        return joblib.load(encoder_path)

    def load_charging_encoder(self):
        encoder_path = (
            self.project_root
            / "dataset"
            / "processed"
            / "charging_encoder.pkl"
        )

        return joblib.load(encoder_path)
    
    def load_shap_explainer(self, random_forest_model):
        """
        Load SHAP TreeExplainer for the trained Random Forest model.
        """

        explainer = shap.TreeExplainer(random_forest_model)

        return explainer