"""FROZEN contract values -- schema v1.0.0.

Everything in this module is part of the team contract (docs/SCHEMA_CONTRACT.md).
Changing a value means: bump SCHEMA_VERSION, update the contract document, and get
sign-off from all three workstream owners.  Teammates should import from here
instead of re-typing names, units, cut-offs or topic names.
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------- variables
TARGET = "target"                       # JSON key / column name
TARGET_SOURCE_COLUMN = "Global_active_power"
TARGET_UNIT = "kW"

FEATURES = (
    "global_reactive_power",            # kW
    "voltage",                          # V
    "global_intensity",                 # A
    "sub_metering_1",                   # Wh per minute (kitchen)
    "sub_metering_2",                   # Wh per minute (laundry)
    "sub_metering_3",                   # Wh per minute (water heater / AC)
)
VALUE_COLUMNS = (TARGET,) + FEATURES

UCI_COLUMN_MAP = {
    "Global_active_power": TARGET,
    "Global_reactive_power": "global_reactive_power",
    "Voltage": "voltage",
    "Global_intensity": "global_intensity",
    "Sub_metering_1": "sub_metering_1",
    "Sub_metering_2": "sub_metering_2",
    "Sub_metering_3": "sub_metering_3",
}
UCI_EXPECTED_ROWS = 2_075_259           # Im et al. (2024), Sec. 5.2

# ------------------------------------------------------------------- time
RAW_INTERVAL_S = 60                     # source sampling interval
FORECAST_INTERVAL_S = 3600              # PREDICTION INTERVAL: 1 hour
FORECAST_INTERVAL_ISO = "PT1H"
FORECAST_HORIZON_STEPS = 1              # one step ahead = the next hour
SEASONAL_PERIOD = 24                    # hours; matches SARIMA (1,0,1,24) in the base paper
MINUTES_PER_WINDOW = FORECAST_INTERVAL_S // RAW_INTERVAL_S   # 60
ALLOWED_LATENESS_S = 300                # event-time lateness before a window closes
TIMEZONE_NOTE = "UCI wall-clock stamps are labelled UTC as-is (no DST adjustment)."

# Chronological split, half-open intervals on event_time.  train+val == the
# base paper's training period (through Dec 2009); test == its 11-month validation period.
SPLIT_TRAIN_END = "2009-10-01T00:00:00Z"   # train: start .. < 2009-10-01
SPLIT_VAL_END = "2010-01-01T00:00:00Z"     # val:   2009-10-01 .. < 2010-01-01
                                           # test:  2010-01-01 .. end of data
SPLITS = ("train", "val", "test")

# --------------------------------------------------------------- identity
PRIMARY_METER_ID = "uci-001"


def meter_id(index: int) -> str:
    """index is 1-based: 1 -> 'uci-001'. Meters > 1 are load replicas of the same series."""
    return f"uci-{index:03d}"


# ----------------------------------------------------------------- kafka
TOPIC_RAW = "energy.raw"
TOPIC_HOURLY = "energy.hourly"          # PROPOSED addition (processed stream)
TOPIC_PREDICTIONS = "energy.predictions"
TOPIC_ANOMALIES = "energy.anomalies"
TOPIC_DRIFT = "energy.drift"
TOPIC_MODEL_EVENTS = "energy.model_events"
TOPIC_METRICS = "energy.metrics"
ALL_TOPICS = (TOPIC_RAW, TOPIC_HOURLY, TOPIC_PREDICTIONS, TOPIC_ANOMALIES,
              TOPIC_DRIFT, TOPIC_MODEL_EVENTS, TOPIC_METRICS)

GROUP_BASE = "telegraf-base"            # Telegraf -> InfluxDB (base pipeline)
GROUP_METRICS = "telegraf-metrics"
GROUP_ADAPTIVE = "adaptive-processor"   # adaptive analytics layer

# --------------------------------------------------------------- influxdb
MEASUREMENT_RAW = "energy_raw"
MEASUREMENT_METRIC = "pipeline_metric"
