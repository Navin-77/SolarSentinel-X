import os
import sys
import time
import serial


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# IMPORTS
# ============================================================

from realtime.sensor_interface import parse_sensor_line
from models.real_inference import RealModelInference


# ============================================================
# SERIAL CONFIGURATION
# ============================================================

PORT = "COM11"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 2


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SOLARSENTINEL-X")
    print("REAL ESP32 → AI LIVE INFERENCE")
    print("=" * 70)

    print("\nSerial Port :", PORT)
    print("Baud Rate   :", BAUD_RATE)

    # ========================================================
    # LOAD AI MODELS
    # ========================================================

    print("\nInitializing AI models...")

    inference = RealModelInference()

    # ========================================================
    # OPEN SERIAL CONNECTION
    # ========================================================

    ser = None

    try:

        print("\nConnecting to ESP32...")

        ser = serial.Serial(
            PORT,
            BAUD_RATE,
            timeout=SERIAL_TIMEOUT
        )

        time.sleep(2)

        print(
            f"Connected to {PORT} @ {BAUD_RATE} baud"
        )

        print("\nWaiting for REAL ESP32 sensor data...")
        print("Press CTRL+C to stop.\n")


        sample_count = 0


        # ====================================================
        # LIVE LOOP
        # ====================================================

        while True:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()


            # ------------------------------------------------
            # Ignore empty lines
            # ------------------------------------------------

            if not line:
                continue


            # ------------------------------------------------
            # Display raw ESP32 line
            # ------------------------------------------------

            print(
                f"\nRAW: {line}"
            )


            # ------------------------------------------------
            # Ignore non-DATA messages
            # ------------------------------------------------

            if not line.startswith("DATA,"):
                continue


            # ------------------------------------------------
            # Parse ESP32 data
            # ------------------------------------------------

            try:

                sensor_data = parse_sensor_line(
                    line
                )

            except Exception as error:

                print(
                    "[PARSE ERROR]:",
                    error
                )

                continue


            if sensor_data is None:

                print(
                    "[WARNING] Sensor data could not be parsed."
                )

                continue


            # ------------------------------------------------
            # Convert parser output to model format
            # ------------------------------------------------

            model_input = {

                "LDR":
                    sensor_data["ldr"],

                "DHT22_Temperature":
                    sensor_data["dht22_temp"],

                "DHT22_Humidity":
                    sensor_data["dht22_humidity"],

                "DS18B20_Temperature":
                    sensor_data["ds18b20_temp"],

                "Solar_Voltage":
                    sensor_data["ina1_voltage"],

                "Solar_Current":
                    sensor_data["ina1_current"],

                "Solar_Power":
                    sensor_data["ina1_power"],

                "Battery_Voltage":
                    sensor_data["ina2_voltage"],

                "Battery_Current":
                    sensor_data["ina2_current"],

                "Battery_Power":
                    sensor_data["ina2_power"],
            }


            # =================================================
            # RUN REAL AI MODELS
            # =================================================

            try:

                result = inference.predict(
                    model_input
                )

            except Exception as error:

                print(
                    "\n[MODEL ERROR]:",
                    error
                )

                continue


            sample_count += 1


            # =================================================
            # DISPLAY SENSOR DATA
            # =================================================

            print("\n" + "-" * 70)

            print(
                f"LIVE SAMPLE #{sample_count}"
            )

            print("-" * 70)

            print(
                f"LDR                 : "
                f"{model_input['LDR']:.2f}"
            )

            print(
                f"DHT22 Temperature   : "
                f"{model_input['DHT22_Temperature']:.2f} °C"
            )

            print(
                f"DHT22 Humidity      : "
                f"{model_input['DHT22_Humidity']:.2f} %"
            )

            print(
                f"DS18B20 Temperature : "
                f"{model_input['DS18B20_Temperature']:.2f} °C"
            )

            print(
                f"Solar Voltage       : "
                f"{model_input['Solar_Voltage']:.3f} V"
            )

            print(
                f"Solar Current       : "
                f"{model_input['Solar_Current']:.3f}"
            )

            print(
                f"Solar Power         : "
                f"{model_input['Solar_Power']:.3f} W"
            )

            print(
                f"Battery Voltage     : "
                f"{model_input['Battery_Voltage']:.3f} V"
            )

            print(
                f"Battery Current     : "
                f"{model_input['Battery_Current']:.3f}"
            )

            print(
                f"Battery Power       : "
                f"{model_input['Battery_Power']:.3f} W"
            )


            # =================================================
            # RANDOM FOREST
            # =================================================

            rf = result["random_forest"]

            print("\n" + "=" * 70)
            print("RANDOM FOREST")
            print("=" * 70)

            print(
                "System Health :",
                rf["prediction"]
            )

            print(
                "Confidence    :",
                f"{rf['confidence'] * 100:.2f}%"
            )

            print("\nClass Probabilities:")

            for label, probability in rf[
                "probabilities"
            ].items():

                print(
                    f"  {label:<10}: "
                    f"{probability * 100:.2f}%"
                )


            # =================================================
            # AUTOENCODER
            # =================================================

            ae = result["autoencoder"]

            print("\n" + "=" * 70)
            print("AUTOENCODER")
            print("=" * 70)

            print(
                "Status               :",
                ae["status"]
            )

            print(
                "Reconstruction Error :",
                f"{ae['reconstruction_error']:.6f}"
            )

            print(
                "Threshold            :",
                f"{ae['threshold']:.6f}"
            )


            # =================================================
            # LSTM
            # =================================================

            lstm = result["lstm"]

            print("\n" + "=" * 70)
            print("LSTM")
            print("=" * 70)

            if lstm["ready"]:

                print(
                    "Prediction :",
                    lstm["prediction"]
                )

                print(
                    "Confidence :",
                    f"{lstm['confidence'] * 100:.2f}%"
                )

            else:

                print(
                    "Status : Waiting for sequence"
                )

                print(
                    "Samples :",
                    f"{lstm['samples_collected']}/"
                    f"{lstm['samples_required']}"
                )


            # =================================================
            # LIVE AI STATUS
            # =================================================

            print("\n" + "=" * 70)
            print("LIVE AI STATUS")
            print("=" * 70)

            print(
                "RF Health      :",
                rf["prediction"]
            )

            print(
                "Anomaly Status  :",
                ae["status"]
            )

            if lstm["ready"]:

                print(
                    "LSTM Prediction:",
                    lstm["prediction"]
                )

            else:

                print(
                    "LSTM Prediction:",
                    "Waiting..."
                )

            print("=" * 70)


    # ========================================================
    # USER STOP
    # ========================================================

    except KeyboardInterrupt:

        print(
            "\n\nStopping live inference..."
        )


    # ========================================================
    # SERIAL ERROR
    # ========================================================

    except serial.SerialException as error:

        print(
            "\nSERIAL CONNECTION ERROR:"
        )

        print(error)

        print(
            f"\nCheck that the ESP32 is connected to {PORT}."
        )

        print(
            "Also make sure Arduino Serial Monitor is CLOSED."
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        if ser is not None:

            try:

                ser.close()

            except Exception:
                pass

        print(
            "\nESP32 serial connection closed."
        )

        print(
            "Live inference stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()