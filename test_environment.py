import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import shap
import plotly
import streamlit

print("=" * 50)
print("SolarSentinel-X Environment Check")
print("=" * 50)

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("TensorFlow:", tf.__version__)
print("SHAP:", shap.__version__)
print("Plotly:", plotly.__version__)
print("Streamlit:", streamlit.__version__)

print("\n✅ All libraries imported successfully!")