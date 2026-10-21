"""Regenerate grafana/dashboards/*.json.  Edit the Flux here, run `python tools/build_dashboards.py`.

Two dashboards because they live on two different time axes:
  energy_readings  -> EVENT time (UCI 2006-2010 timestamps).  Set the Grafana time range to the replayed period.
  pipeline_health  -> WALL-CLOCK time (throughput, lag, latency, CPU, memory).  Default: last 15 minutes.
"""
import json
from pathlib import Path

DS = {"type": "influxdb", "uid": "influxdb-energy"}
B = "energy"          # bucket name; keep in sync with INFLUX_BUCKET
OUT = Path(__file__).resolve().parent.parent / "grafana" / "dashboards"


def target(flux, ref="A"):
    return {"refId": ref, "datasource": DS, "query": flux}


def ts(pid, title, flux_list, x, y, w=12, h=8, unit="short", desc="", time_from=None):
    p = {"id": pid, "type": "timeseries", "title": title, "description": desc, "datasource": DS,
         "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "targets": [target(f, chr(65 + i)) for i, f in enumerate(flux_list)],
         "fieldConfig": {"defaults": {"unit": unit, "custom": {"drawStyle": "line", "lineWidth": 1, "fillOpacity": 8,
                                                                 "showPoints": "never", "spanNulls": True}},
                         "overrides": []},
         "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}}
    if time_from:
        p["timeFrom"] = time_from
    return p


def stat(pid, title, flux, x, y, w=6, h=4, unit="short", desc=""):
    return {"id": pid, "type": "stat", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": [target(flux)],
            "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": "value", "graphMode": "none"}}


def text(pid, title, md, x, y, w=24, h=4):
    return {"id": pid, "type": "text", "title": title, "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "options": {"mode": "markdown", "content": md}}


def var_query(name, label, flux, include_all=False):
    v = {"name": name, "label": label, "type": "query", "datasource": DS, "query": flux, "refresh": 2,
         "sort": 1, "multi": False, "includeAll": include_all, "current": {}}
    if include_all:
        v["allValue"] = ".*"; v["current"] = {"text": "All", "value": "$__all"}
    return v


def dash(uid, title, panels, variables, time_from, time_to, refresh):
    return {"uid": uid, "title": title, "schemaVersion": 39, "version": 1, "editable": True, "tags": ["capstone"],
            "timezone": "utc", "refresh": refresh, "time": {"from": time_from, "to": time_to},
            "templating": {"list": variables}, "annotations": {"list": []}, "panels": panels}


RAW = f'r._measurement == "energy_raw" and r.run_id == "${{run_id}}" and r.meter_id == "${{meter_id}}"'


def raw_series(fields, fn="mean"):
    cond = " or ".join(f'r._field == "{f}"' for f in fields)
    return (f'from(bucket: "{B}")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
            f'  |> filter(fn: (r) => {RAW})\n  |> filter(fn: (r) => {cond})\n'
            f'  |> aggregateWindow(every: v.windowPeriod, fn: {fn}, createEmpty: false)')


readings = [
    text(1, "How to read this dashboard",
         "X-axis is **event time** (the UCI timestamp), not wall-clock. Set the time range (top right) to the "
         "replayed period, e.g. `2010-01-01` to `2010-02-01`. Prediction, anomaly, drift and model-selection "
         "panels are added once the forecasting / detection owners publish their output schemas.", 0, 0, 24, 3),
    ts(2, "Global active power (target, kW)", [raw_series(["target"])], 0, 3, 24, 9, "kwatt"),
    ts(3, "Voltage (V)", [raw_series(["voltage"])], 0, 12, 8, 7, "volt"),
    ts(4, "Global intensity (A)", [raw_series(["global_intensity"])], 8, 12, 8, 7, "amp"),
    ts(5, "Global reactive power (kW)", [raw_series(["global_reactive_power"])], 16, 12, 8, 7, "kwatt"),
    ts(6, "Sub-metering (Wh per minute)", [raw_series(["sub_metering_1"]), raw_series(["sub_metering_2"]),
                                          raw_series(["sub_metering_3"])], 0, 19, 24, 7, "short"),
    stat(7, "Readings stored (range)",
         f'from(bucket: "{B}")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
         f'  |> filter(fn: (r) => {RAW} and r._field == "seq")\n  |> group()\n  |> count()', 0, 26, 6),
    stat(8, "Missing minutes (range)",
         f'from(bucket: "{B}")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
         f'  |> filter(fn: (r) => {RAW} and r._field == "is_missing")\n  |> toInt()\n  |> group()\n  |> sum()',
         6, 26, 6, desc="Minutes whose target reading was '?' in the source (kept as is_missing=true)."),
    stat(9, "Mean power (range, kW)",
         f'from(bucket: "{B}")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
         f'  |> filter(fn: (r) => {RAW} and r._field == "target")\n  |> group()\n  |> mean()', 12, 26, 6, unit="kwatt"),
]
vars_readings = [
    var_query("run_id", "Run",
              'import "influxdata/influxdb/schema"\nschema.tagValues(bucket: "energy", tag: "run_id", '
              'predicate: (r) => r._measurement == "energy_raw", start: 2006-01-01T00:00:00Z)'),
    var_query("meter_id", "Meter",
              'import "influxdata/influxdb/schema"\nschema.tagValues(bucket: "energy", tag: "meter_id", '
              'predicate: (r) => r._measurement == "energy_raw" and r.run_id == "${run_id}", start: 2006-01-01T00:00:00Z)'),
]


def metric(name, extra="", fn="mean", every="v.windowPeriod"):
    return (f'from(bucket: "{B}")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
            f'  |> filter(fn: (r) => r._measurement == "pipeline_metric" and r.name == "{name}" '
            f'and r.run_id =~ /${{run_id:regex}}/)\n  |> aggregateWindow(every: {every}, fn: {fn}, createEmpty: false)'
            f'{extra}\n  |> group(columns: ["instance", "_field"])')


health = [
    ts(1, "Producer rate (records/s)", [metric("producer.rate_rps")], 0, 0, 12, 8, "short",
       "Achieved emission rate of the replay producer vs. the configured load profile."),
    ts(2, "Telegraf -> InfluxDB points written/s",
       [f'from(bucket: "{B}")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
        f'  |> filter(fn: (r) => r._measurement == "internal_write" and r._field == "metrics_written" and r.output == "influxdb_v2")\n'
        f'  |> derivative(unit: 1s, nonNegative: true)\n  |> group(columns: ["_field"])'], 12, 0, 12, 8, "short"),
    ts(3, "Kafka consumer lag (records)", [metric("kafka.consumer_lag", fn="max")], 0, 8, 12, 8, "short",
       "Records the group is behind the log end. Sustained growth = the consumer cannot keep up."),
    ts(4, "End-to-end latency: producer -> queryable in InfluxDB (ms)",
       [metric("e2e.latency_ms", fn="mean"), metric("e2e.latency_ms", fn="max").replace('"mean"', '"max"')],
       12, 8, 12, 8, "ms", "Sampled by cap-probe; resolution = poll interval (default 200 ms)."),
    ts(5, "CPU per container (% of one core)", [metric("docker.cpu_percent")], 0, 16, 12, 8, "percent"),
    ts(6, "Memory per container (MiB)", [metric("docker.mem_mib")], 12, 16, 12, 8, "short"),
    ts(7, "Telegraf output buffer (metrics)",
       [f'from(bucket: "{B}")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'
        f'  |> filter(fn: (r) => r._measurement == "internal_write" and r._field == "buffer_size" and r.output == "influxdb_v2")\n'
        f'  |> group(columns: ["_field"])'], 0, 24, 12, 7, "short",
       "Growth here means InfluxDB writes are slower than ingestion."),
]
vars_health = [var_query("run_id", "Run",
                         'import "influxdata/influxdb/schema"\nschema.tagValues(bucket: "energy", tag: "run_id", '
                         'predicate: (r) => r._measurement == "pipeline_metric", start: -30d)', include_all=True)]

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "energy_readings.json").write_text(json.dumps(
    dash("energy-readings", "Energy - Readings (event time)", readings, vars_readings,
         "2010-01-01T00:00:00.000Z", "2010-02-01T00:00:00.000Z", ""), indent=1))
(OUT / "pipeline_health.json").write_text(json.dumps(
    dash("pipeline-health", "Pipeline - Health (wall-clock)", health, vars_health, "now-15m", "now", "5s"), indent=1))
print("wrote dashboards to", OUT)
