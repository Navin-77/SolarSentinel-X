import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# -----------------------------
# DATASET CONFIGURATION
# -----------------------------

NUM_SAMPLES = 10000

START_TIME = datetime(2026, 1, 1, 6, 0, 0)

OUTPUT_FOLDER = "dataset/synthetic"

OUTPUT_FILE = "solar_microgrid_dataset.csv"

# -----------------------------
# BATTERY CONFIGURATION
# -----------------------------

BATTERY_CAPACITY = 100        # Ah

BATTERY_FULL_VOLTAGE = 14.4

BATTERY_EMPTY_VOLTAGE = 11.8

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# LOAD CONFIGURATION
# -----------------------------

MAX_LOAD_POWER = 150      # Watts

def simulate_weather(hour):
    """
    Simulate realistic weather conditions
    """

    # Daylight from 6 AM to 6 PM
    if 6 <= hour <= 18:

        irradiance = max(
            0,
            1000 * np.sin(np.pi * (hour - 6) / 12)
            + random.uniform(-40, 40)
        )

    else:

        irradiance = 0.0

    ambient_temp = random.uniform(22, 38)

    humidity = random.uniform(35, 85)

    return irradiance, ambient_temp, humidity

def simulate_panel(irradiance, ambient_temp):
    """
    Simulate solar panel behaviour
    """

    panel_current = (irradiance / 1000) * 11

    panel_voltage = 17.5 + (irradiance / 1000)

    panel_power = panel_voltage * panel_current

    panel_temperature = ambient_temp + (irradiance / 1000) * 20

    return (
        panel_voltage,
        panel_current,
        panel_power,
        panel_temperature
    )
    
def simulate_mppt(panel_voltage, panel_power):
    """
    Simulate MPPT charge controller
    """

    mppt_efficiency = random.uniform(0.96, 0.98)

    mppt_output_voltage = 14.4

    mppt_input_voltage = panel_voltage

    output_power = panel_power * mppt_efficiency

    return (
        mppt_input_voltage,
        mppt_output_voltage,
        mppt_efficiency,
        output_power
    )    
    
def simulate_battery(power_balance):
    """
    Simulate an independent battery state for each sample
    """

    # -------------------------------------------------
    # Generate Battery State of Charge (SOC)
    # -------------------------------------------------

    soc_probability = random.random()

    if soc_probability < 0.75:
        battery_soc = random.uniform(60, 95)

    elif soc_probability < 0.90:
        battery_soc = random.uniform(30, 60)

    else:
        battery_soc = random.uniform(10, 30)

    # -------------------------------------------------
    # Generate Battery State of Health (SOH)
    # -------------------------------------------------

    soh_probability = random.random()

    if soh_probability < 0.90:
        battery_soh = random.uniform(90, 98)

    else:
        battery_soh = random.uniform(80, 90)

    # -------------------------------------------------
    # Battery Voltage based on SOC
    # -------------------------------------------------

    battery_voltage = (
        BATTERY_EMPTY_VOLTAGE
        +
        (battery_soc / 100)
        *
        (BATTERY_FULL_VOLTAGE - BATTERY_EMPTY_VOLTAGE)
    )

    # -------------------------------------------------
    # Battery Current
    # -------------------------------------------------

    battery_current = power_balance / BATTERY_FULL_VOLTAGE

    # -------------------------------------------------
    # Battery Temperature
    # -------------------------------------------------

    battery_temperature = (
        28
        +
        abs(battery_current) * 0.8
        +
        random.uniform(-1, 1)
    )

    # -------------------------------------------------
    # Charging Status
    # -------------------------------------------------

    if power_balance >= 0:
        charging_status = "Charging"
    else:
        charging_status = "Discharging"

    return (
        battery_voltage,
        battery_current,
        battery_soc,
        battery_soh,
        battery_temperature,
        charging_status
    )
    
# -------------------------------------------------
# Fault Injection Engine
# -------------------------------------------------

def inject_fault():

    probability = random.random()

    if probability < 0.75:
        return "No Fault"

    elif probability < 0.83:
        return "Low Battery SOC"

    elif probability < 0.89:
        return "MPPT Efficiency Drop"

    elif probability < 0.94:
        return "Low Solar Irradiance"

    elif probability < 0.98:
        return "Minor System Degradation"

    else:
        return "Battery Overheating"
       

def simulate_load(hour):

    """
    Simulate electrical load
    """

    if 6 <= hour <= 9:

        load_power = random.uniform(40, 70)

    elif 10 <= hour <= 16:

        load_power = random.uniform(80, 120)

    elif 17 <= hour <= 22:

        load_power = random.uniform(100, 150)

    else:

        load_power = random.uniform(20, 40)

    load_current = load_power / 12

    return (
        load_power,
        load_current
    )

# -----------------------------
# CREATE DATASET
# -----------------------------

dataset = []    
    
current_time = START_TIME

for i in range(NUM_SAMPLES):

    hour = current_time.hour

    irr, temp, hum = simulate_weather(hour)

    voltage, current, power, panel_temp = simulate_panel(
        irr,
        temp
    )
    
    (
        mppt_in,
        mppt_out,
        efficiency,
        output_power
    ) = simulate_mppt(
        voltage,
        power
    )
    
    load_power, load_current = simulate_load(hour)

    power_balance = output_power - load_power
    
    fault_type = inject_fault()

    (
        battery_voltage,
        battery_current,
        battery_soc,
        battery_soh,
        battery_temperature,
        charging_status
    ) = simulate_battery(
        power_balance
    )
    
    # -------------------------------------------------
    # Apply Fault Effects
    # -------------------------------------------------

    if fault_type == "Low Battery SOC":

        battery_soc = random.uniform(10, 25)

        battery_voltage = (
            BATTERY_EMPTY_VOLTAGE
            +
            (battery_soc / 100)
            *
            (BATTERY_FULL_VOLTAGE - BATTERY_EMPTY_VOLTAGE)
        )

    elif fault_type == "MPPT Efficiency Drop":

        efficiency = random.uniform(0.85, 0.93)

        output_power = power * efficiency

        power_balance = output_power - load_power

    elif fault_type == "Low Solar Irradiance":

        if 6 <= hour <= 18:

            irr = random.uniform(20, 100)

            voltage, current, power, panel_temp = simulate_panel(
                irr,
                temp
            )

            output_power = power * efficiency

            power_balance = output_power - load_power

    elif fault_type == "Battery Overheating":

        battery_temperature = random.uniform(50, 60)

    elif fault_type == "Minor System Degradation":

        battery_soh = random.uniform(80, 88)

        battery_temperature += random.uniform(2, 4)
    
    efficiency_loss = (1 - efficiency) * 100
    
    # -----------------------------
    # SYSTEM HEALTH
    # -----------------------------

    health_score = 100

    # Battery SOC contribution
    if battery_soc < 30:
        health_score -= 35
    elif battery_soc < 60:
        health_score -= 15

    # Battery Temperature contribution
    if battery_temperature >= 50:
        health_score -= 35
    elif battery_temperature >= 40:
        health_score -= 15

    # MPPT Efficiency contribution
    if efficiency < 0.90:
        health_score -= 20
    elif efficiency < 0.95:
        health_score -= 10

    # Battery SOH contribution
    if battery_soh < 85:
        health_score -= 20
    elif battery_soh < 90:
        health_score -= 10

    # Final Health Classification
    if health_score >= 80:
        system_health = "Healthy"

    elif health_score >= 50:
        system_health = "Warning"

    else:
        system_health = "Critical"

        
    battery_stress_index = (
        (battery_temperature * 0.4)
        +
        (abs(power_balance) * 0.05)
        +
        ((100 - battery_soc) * 0.2)
    )
    
    # -------------------------------------------------
    # Remaining Useful Life (RUL)
    # -------------------------------------------------

    battery_rul = max(
        0,
        ((battery_soh - 80) / 20) * 5000
    )

    dataset.append({

        "Timestamp": current_time,

        "Hour": hour,
        
        "Day": current_time.day,

        "Solar_Irradiance": round(irr, 2),

        "Ambient_Temperature": round(temp, 2),

        "Humidity": round(hum, 2),

        "Panel_Voltage": round(voltage, 2),

        "Panel_Current": round(current, 2),

        "Panel_Power": round(power, 2),

        "Panel_Temperature": round(panel_temp, 2),

        "MPPT_Input_Voltage": round(mppt_in, 2),

        "MPPT_Output_Voltage": round(mppt_out, 2),

        "MPPT_Efficiency": round(efficiency, 4),

        "Battery_Voltage": round(battery_voltage, 2),

        "Battery_Current": round(battery_current, 2),

        "Battery_SOC": round(battery_soc, 2),

        "Battery_SOH": round(battery_soh, 4),
        
        "Battery_RUL": round(battery_rul, 2),

        "Battery_Temperature": round(battery_temperature, 2),

        "Charging_Status": charging_status,

        "Load_Power": round(load_power, 2),

        "Load_Current": round(load_current, 2),

        "Power_Balance": round(power_balance, 2),
        
        "Battery_Stress_Index": round(battery_stress_index, 2),
        
        "Efficiency_Loss": round(efficiency_loss, 2),
        
        "Health_Score": round(health_score, 2),
        
        "System_Health": system_health,
        
        "Fault_Type": fault_type

    })

    current_time += timedelta(hours=1)
    
df = pd.DataFrame(dataset)

output_path = os.path.join(
    OUTPUT_FOLDER,
    OUTPUT_FILE
)

df.to_csv(
    output_path,
    index=False
)

print("Dataset Generated Successfully!")
print(f"Total Samples : {len(df)}")
print(f"Saved To : {output_path}")