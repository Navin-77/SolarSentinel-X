from digital_twin.model_loader import ModelLoader

loader = ModelLoader()

print("Loading Random Forest...")
loader.load_random_forest()
print("✓ Random Forest Loaded")

print("Loading LSTM...")
loader.load_lstm()
print("✓ LSTM Loaded")

print("Loading Autoencoder...")
loader.load_autoencoder()
print("✓ Autoencoder Loaded")

print("Loading Threshold...")
print(loader.load_threshold())

print("Loading Scaler...")
loader.load_scaler()
print("✓ Scaler Loaded")

print("Loading Health Encoder...")
loader.load_health_encoder()
print("✓ Health Encoder Loaded")

print("Loading Charging Encoder...")
loader.load_charging_encoder()
print("✓ Charging Encoder Loaded")

print("\nAll models loaded successfully.")