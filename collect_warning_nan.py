import csv, math, time
from datetime import datetime
from pathlib import Path
import serial
from realtime.sensor_interface import parse_sensor_line

PORT = "COM11"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 2
TARGET = 100
OUTPUT_FILE = Path(__file__).resolve().parent / "dataset" / "real_training" / "warning_nan_dataset.csv"

FEATURES = [
    "LDR","DHT22_Temperature","DHT22_Humidity","DS18B20_Temperature",
    "Solar_Voltage","Solar_Current","Solar_Power",
    "Battery_Voltage","Battery_Current","Battery_Power"
]
COLUMNS = ["Timestamp", *FEATURES, "System_Health"]

def convert(p):
    return {
        "LDR": p["ldr"], "DHT22_Temperature": p["dht22_temp"],
        "DHT22_Humidity": p["dht22_humidity"],
        "DS18B20_Temperature": p["ds18b20_temp"],
        "Solar_Voltage": p["ina1_voltage"], "Solar_Current": p["ina1_current"],
        "Solar_Power": p["ina1_power"], "Battery_Voltage": p["ina2_voltage"],
        "Battery_Current": p["ina2_current"], "Battery_Power": p["ina2_power"]
    }

def valid(d):
    for f in FEATURES:
        if f == "DS18B20_Temperature": continue
        try: v = float(d[f])
        except: return False, f"Non-numeric: {f}"
        if not math.isfinite(v): return False, f"Invalid: {f}"
    h = float(d["DHT22_Humidity"])
    if float(d["DHT22_Temperature"]) == 0 and h == 0: return False, "DHT22 0/0"
    if not 0 <= h <= 100: return False, "Invalid humidity"
    if not 0 <= float(d["LDR"]) <= 4095: return False, "Invalid LDR"
    try:
        x = float(d["DS18B20_Temperature"])
        if not math.isfinite(x): raise ValueError
    except:
        d["DS18B20_Temperature"] = float("nan")
    return True, "Valid"

def count():
    if not OUTPUT_FILE.exists(): return 0
    with open(OUTPUT_FILE, newline="", encoding="utf-8") as f:
        return sum(1 for r in csv.DictReader(f) if r.get("System_Health") == "Warning")

def append(d):
    row = {"Timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
           "System_Health": "Warning"}
    for f in FEATURES:
        v = d[f]
        if f == "DS18B20_Temperature":
            try:
                v = float(v)
                if not math.isfinite(v): v = "NaN"
            except: v = "NaN"
        row[f] = v
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    existing = count()
    print("="*70)
    print("SolarSentinel-X WARNING NaN DATA COLLECTION")
    print("="*70)
    print(f"Existing Warning: {existing}/100")
    print("DS18B20 NaN is accepted; all other sensors must be valid.")
    print("This creates a SEPARATE temporary CSV.")
    if existing >= TARGET: return

    input("\nPut system in intended Warning condition, close Arduino Serial Monitor, then press ENTER...")

    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
    except serial.SerialException as e:
        print(f"Could not open {PORT}: {e}")
        return

    print(f"Connected to {PORT}. Collecting...")
    collected = 0
    try:
        while existing + collected < TARGET:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("DATA,"): continue
            try:
                p = parse_sensor_line(line)
                if p is None: raise ValueError("parser returned None")
                d = convert(p)
            except Exception as e:
                print("[REJECTED DATA]", e); continue

            ok, reason = valid(d)
            if not ok:
                print("[REJECTED]", reason); continue

            append(d)
            collected += 1

            # Show EVERY real sensor value so the physical condition
            # can be checked before accepting the sample.
            print("\n" + "-" * 75)
            print(f"ACCEPTED WARNING SAMPLE {existing + collected}/100")
            print("-" * 75)
            print(f"LDR                 : {float(d['LDR']):.2f}")
            print(f"DHT22 Temperature   : {float(d['DHT22_Temperature']):.2f} C")
            print(f"DHT22 Humidity      : {float(d['DHT22_Humidity']):.2f} %")
            print(f"DS18B20 Temperature : NaN  <-- SENSOR UNAVAILABLE")
            print(f"Solar Voltage       : {float(d['Solar_Voltage']):.4f} V")
            print(f"Solar Current       : {float(d['Solar_Current']):.6f}")
            print(f"Solar Power         : {float(d['Solar_Power']):.6f}")
            print(f"Battery Voltage     : {float(d['Battery_Voltage']):.4f} V")
            print(f"Battery Current     : {float(d['Battery_Current']):.6f}")
            print(f"Battery Power       : {float(d['Battery_Power']):.6f}")
            print("-" * 75)
            print("ALL OTHER VALUES ABOVE ARE REAL ESP32 READINGS.")
            print("ONLY DS18B20 IS ALLOWED TO BE NaN.")

    except KeyboardInterrupt:
        print("\nCollection stopped. Accepted this session:", collected)
    finally:
        ser.close()

    print(f"\nFinal Warning count: {count()}/100")
    print("Saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
