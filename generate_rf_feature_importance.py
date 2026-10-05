import matplotlib.pyplot as plt

features = [
    "Solar_Power",
    "DHT22_Humidity",
    "DHT22_Temperature",
    "LDR",
    "Solar_Voltage",
    "Solar_Current",
    "Battery_Voltage",
    "DS18B20_Temperature",
    "Battery_Current",
    "Battery_Power"
]

importance = [
    0.162706,
    0.155022,
    0.138640,
    0.137567,
    0.132428,
    0.111749,
    0.080990,
    0.080897,
    0.000000,
    0.000000
]

plt.figure(figsize=(10, 6))

plt.barh(features[::-1], importance[::-1])

plt.xlabel("Feature Importance")
plt.ylabel("Sensor Feature")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "results/random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure 8 generated successfully!")
print("Saved as: results/random_forest_feature_importance.png")