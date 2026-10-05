from digital_twin.utils import load_latest_sensor_data

sensor_data = load_latest_sensor_data()

print("\nSequence Shape:")
print(sensor_data.shape)

print("\nLast 5 Rows:")
print(sensor_data.tail())