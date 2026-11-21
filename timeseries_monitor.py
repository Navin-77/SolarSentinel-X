# -*- coding: utf-8 -*-
"""
Timeseries monitoring utilities for SolarSentinel‑X.

- Every 60 seconds a sensor snapshot is validated (via
  ``RealModelInference.validate_sensor_data``) and written to
  ``logs/timeseries_1min_log.csv`` together with the decision engine
  outcome (``NORMAL``, ``WARNING``, ``CRITICAL`` etc.).
- Every 600 seconds a lightweight check determines whether a sufficient
  number of *new* valid samples have been collected since the previous
  training run.  If the threshold is met, the full real‑pipeline is
  re‑trained by invoking ``retrain_all_real.py``.  The attempt –
  timestamp, number of candidate samples and the result – is appended to
  ``logs/retrain_10min_history.csv``.

The module is deliberately self‑contained: it does **not** import or
modify any of the existing production code (steps 5‑8).  It can be wired
into the live system (e.g. from ``realtime/real_system_controller.py``) or
run as a stand‑alone background process.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path
import sys
from subprocess import run, PIPE
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Helper: ensure the ``logs`` directory exists.
# ---------------------------------------------------------------------------
LOGS_DIR = Path(__file__).resolve().parents[0] / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Paths for the two CSV logs.
TIMESERIES_LOG = LOGS_DIR / "timeseries_1min_log.csv"
RETRAIN_LOG = LOGS_DIR / "retrain_10min_history.csv"

# ---------------------------------------------------------------------------
# Validation & decision handling utilities
# ---------------------------------------------------------------------------
# Import the real inference class lazily to avoid heavy imports at module
# load time.  The class lives in ``models/real_inference.py``.
from models.real_inference import RealModelInference

# Feature list – kept in sync with the project’s 10‑feature schema.
FEATURE_NAMES = [
    "LDR",
    "DHT22_Temperature",
    "DHT22_Humidity",
    "DS18B20_Temperature",
    "Solar_Voltage",
    "Solar_Current",
    "Solar_Power",
    "Battery_Voltage",
    "Battery_Current",
    "Battery_Power",
]


def _is_night_standby(sensor: Dict[str, float]) -> bool:
    """Return ``True`` if the reading corresponds to a valid low‑light /
    night‑standby situation.

    Criteria (mirroring Step 5 logic):
    * LDR > 1100 (low light)
    * Solar_Power == 0
    * Battery_Voltage within a healthy range (3 V – 15 V)
    """
    try:
        ldr = float(sensor.get("LDR", 0))
        solar_power = float(sensor.get("Solar_Power", 0))
        batt = float(sensor.get("Battery_Voltage", 0))
    except (TypeError, ValueError):
        return False
    return ldr > 1100 and solar_power == 0 and 3.0 <= batt <= 15.0


def _validate_sensor(sensor: Dict) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """Validate a raw sensor dictionary.

    Returns a tuple ``(is_valid, error_msg, validated_dict)`` where:
    * ``is_valid`` – ``True`` when no fault was detected.
    * ``error_msg`` – ``None`` if valid, otherwise the ``ValueError`` text.
    * ``validated_dict`` – the dictionary after ``DS18B20_Temperature`` has
      been derived (or ``None`` on error).
    """
    try:
        validated = RealModelInference().validate_sensor_data(sensor)
        # ``validate_sensor_data`` returns a NumPy array of values ordered
        # according to ``FEATURE_NAMES``.  Convert back to a dict for CSV.
        validated_dict = {name: float(val) for name, val in zip(FEATURE_NAMES, validated)}
        return True, None, validated_dict
    except ValueError as err:
        return False, str(err), None

# ---------------------------------------------------------------------------
# Timeseries recorder (1‑minute interval)
# ---------------------------------------------------------------------------
class TimeseriesRecorder(threading.Thread):
    """Background thread that records a validated snapshot every minute.

    Parameters
    ----------
    get_sensor_data: Callable[[], Dict]
        Function returning the current raw sensor dict.
    get_decision_status: Callable[[], str]
        Function returning the latest decision‑engine overall status
        (e.g. ``"NORMAL"``).  If the sensor is faulty the status will be
        overridden to ``"CRITICAL"`` automatically.
    """

    def __init__(self, get_sensor_data, get_decision_status):
        super().__init__(daemon=True)
        self.get_sensor_data = get_sensor_data
        self.get_decision_status = get_decision_status
        self._stop_event = threading.Event()
        # Ensure the CSV header exists.
        if not TIMESERIES_LOG.exists():
            with TIMESERIES_LOG.open("w", newline="") as f:
                writer = csv.writer(f)
                header = ["timestamp"] + FEATURE_NAMES + ["decision_status", "error_msg"]
                writer.writerow(header)

    def run(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()
            raw = self.get_sensor_data()
            is_valid, err_msg, validated = _validate_sensor(raw)
            timestamp = datetime.utcnow().isoformat()
            # Determine the decision status to store.
            if not is_valid:
                decision = "CRITICAL"
            elif _is_night_standby(raw):
                decision = "NORMAL"
            else:
                decision = self.get_decision_status()

            row = [timestamp]
            if validated:
                row.extend([validated.get(name, "") for name in FEATURE_NAMES])
            else:
                # Fill placeholders when validation failed.
                row.extend(["" for _ in FEATURE_NAMES])
            row.append(decision)
            row.append(err_msg or "")
            with TIMESERIES_LOG.open("a", newline="") as f:
                csv.writer(f).writerow(row)

            # Sleep the remainder of the minute.
            elapsed = time.time() - start
            wait = max(0, 60 - elapsed)
            self._stop_event.wait(wait)

    def stop(self):
        self._stop_event.set()

# ---------------------------------------------------------------------------
# Retraining watchdog (10‑minute interval)
# ---------------------------------------------------------------------------
class RetrainWatchdog(threading.Thread):
    """Background thread that checks for enough new samples and triggers
    a full retrain via ``retrain_all_real.py``.

    The watchdog keeps a tiny state file (``.last_retrain``) storing the
    ISO timestamp of the most recent successful training run.  Only rows
    newer than that timestamp are considered *new* samples.
    """

    MIN_NEW_SAMPLES = 100  # Minimum records required to justify a retrain.
    STATE_FILE = LOGS_DIR / ".last_retrain"

    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        # Initialise history CSV.
        if not RETRAIN_LOG.exists():
            with RETRAIN_LOG.open("w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "new_samples", "outcome", "details"])
        # Ensure state file exists.
        if not self.STATE_FILE.exists():
            self.STATE_FILE.write_text(datetime.min.isoformat())

    def _read_last_retrain(self) -> datetime:
        try:
            ts = self.STATE_FILE.read_text().strip()
            return datetime.fromisoformat(ts)
        except Exception:
            return datetime.min

    def _write_last_retrain(self, ts: datetime) -> None:
        self.STATE_FILE.write_text(ts.isoformat())

    def run(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()
            try:
                last_rt = self._read_last_retrain()
                # Load timeseries log and count rows newer than last_rt.
                new_rows: List[Dict] = []
                with TIMESERIES_LOG.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_ts = datetime.fromisoformat(row["timestamp"])
                        if row_ts > last_rt and not row["error_msg"]:
                            # Only count rows without validation errors.
                            new_rows.append(row)
                n_new = len(new_rows)
                outcome = "SKIPPED"
                details = "Not enough new valid samples"
                if n_new >= self.MIN_NEW_SAMPLES:
                    # Trigger a fresh retrain – we run the script as a subprocess
                    # to keep the isolated environment (no import side‑effects).
                    proc = run([sys.executable, "retrain_all_real.py"], cwd=Path(__file__).resolve().parents[0],
                               stdout=PIPE, stderr=PIPE, text=True)
                    if proc.returncode == 0:
                        outcome = "SUCCESS"
                        details = f"Retrained with {n_new} new samples"
                        # Update the last‑retrain timestamp to now.
                        self._write_last_retrain(datetime.utcnow())
                    else:
                        outcome = "FAILED"
                        details = proc.stderr.strip() or "Unknown error"
                # Log the check.
                with RETRAIN_LOG.open("a", newline="") as f:
                    csv.writer(f).writerow([datetime.utcnow().isoformat(), n_new, outcome, details])
            except Exception as exc:  # pragma: no cover – safeguard
                with RETRAIN_LOG.open("a", newline="") as f:
                    csv.writer(f).writerow([datetime.utcnow().isoformat(), 0, "ERROR", str(exc)])
            # Sleep until the next 10‑minute window.
            elapsed = time.time() - start
            wait = max(0, 600 - elapsed)
            self._stop_event.wait(wait)

    def stop(self):
        self._stop_event.set()

# ---------------------------------------------------------------------------
# Convenience starter – can be imported and used from the main controller.
# ---------------------------------------------------------------------------
def start_timeseries_monitor(get_sensor_data, get_decision_status) -> Tuple[TimeseriesRecorder, RetrainWatchdog]:
    """Start both the 1‑minute recorder and the 10‑minute watchdog.

    Returns the two thread objects so the caller can keep references and
    stop them gracefully when the application shuts down.
    """
    recorder = TimeseriesRecorder(get_sensor_data, get_decision_status)
    watchdog = RetrainWatchdog()
    recorder.start()
    watchdog.start()
    return recorder, watchdog

# ---------------------------------------------------------------------------
# Example usage (uncomment for a quick sanity test).
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Dummy sensor provider – replace with real hardware interface.
#     def dummy_sensor():
#         return {
#             "LDR": 1200,
#             "DHT22_Temperature": 22.5,
#             "DHT22_Humidity": 55.0,
#             "Solar_Voltage": 0.0,
#             "Solar_Current": 0.0,
#             "Solar_Power": 0.0,
#             "Battery_Voltage": 12.5,
#             "Battery_Current": 1.0,
#             "Battery_Power": 12.5,
#         }
#
#     def dummy_decision():
#         return "NORMAL"
#
#     rec, wd = start_timeseries_monitor(dummy_sensor, dummy_decision)
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         rec.stop()
#         wd.stop()
