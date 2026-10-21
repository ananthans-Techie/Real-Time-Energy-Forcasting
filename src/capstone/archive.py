"""cap-archive: hot -> cold.  Move event-time-old readings from InfluxDB to a Parquet archive.

"Old" is defined on EVENT time relative to the newest stored reading of the run (the stream
watermark), NOT on wall-clock -- the replayed data is from 2006-2010, so a wall-clock
threshold would archive everything immediately.

  cap-archive --run-id A_ramp --hot-days 30 --out data/archive            # copy only
  cap-archive --run-id A_ramp --hot-days 30 --out data/archive --prune    # copy, verify counts, delete from Influx

Layout (Hive style, readable by pandas / DuckDB / Spark / Grafana-DuckDB):
  <out>/measurement=energy_raw/run_id=<id>/meter_id=<m>/date=YYYY-MM-DD/part.parquet
Re-running is idempotent (a day's file is overwritten with the same rows).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import constants as C
from .env import influx_settings, load_env

FLUX_DAY = '''
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r._measurement == "{meas}" and r.run_id == "{run}")
  |> pivot(rowKey: ["_time", "meter_id"], columnKey: ["_field"], valueColumn: "_value")
  |> drop(columns: ["_start", "_stop", "_measurement", "result", "table"])
'''
FLUX_WATERMARK = '''
from(bucket: "{bucket}")
  |> range(start: 2000-01-01T00:00:00Z, stop: 2100-01-01T00:00:00Z)
  |> filter(fn: (r) => r._measurement == "{meas}" and r.run_id == "{run}" and r._field == "seq")
  |> last() |> group() |> max(column: "_time")
'''


def tidy(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Raw Influx pivot frame -> contract column names/order."""
    df = df.rename(columns={"_time": "event_time"})
    df["run_id"] = run_id
    cols = ["event_time", "run_id", "meter_id", "seq", "replay_time_ms", "is_missing",
            *C.VALUE_COLUMNS]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA                       # target/features are absent on missing minutes
    return df[cols].sort_values(["meter_id", "event_time"]).reset_index(drop=True)


def write_partitions(df: pd.DataFrame, root: Path, run_id: str) -> int:
    """Write one parquet file per (meter, date). Returns rows written."""
    n = 0
    df = df.copy()
    df["_date"] = pd.to_datetime(df["event_time"], utc=True).dt.strftime("%Y-%m-%d")
    for (meter, day), g in df.groupby(["meter_id", "_date"]):
        d = root / "measurement=energy_raw" / f"run_id={run_id}" / f"meter_id={meter}" / f"date={day}"
        d.mkdir(parents=True, exist_ok=True)
        g.drop(columns="_date").to_parquet(d / "part.parquet", engine="pyarrow", compression="zstd", index=False)
        n += len(g)
    return n


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--hot-days", type=int, default=30, help="event-time days to keep hot in InfluxDB")
    ap.add_argument("--out", default="data/archive")
    ap.add_argument("--prune", action="store_true", help="delete archived range from InfluxDB after verification")
    a = ap.parse_args(argv)

    from influxdb_client import InfluxDBClient
    cfg = influx_settings()
    cl = InfluxDBClient(url=cfg["url"], token=cfg["token"], org=cfg["org"], timeout=120_000)
    q = cl.query_api()

    wm_tables = q.query(FLUX_WATERMARK.format(bucket=cfg["bucket"], meas=C.MEASUREMENT_RAW, run=a.run_id))
    wm = next((r.values["_time"] for t in wm_tables for r in t.records), None)
    if wm is None:
        print("No data for that run_id in InfluxDB."); return 1
    cutoff = (wm - timedelta(days=a.hot_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    first_q = q.query(f'from(bucket:"{cfg["bucket"]}") |> range(start: 2000-01-01T00:00:00Z) '
                      f'|> filter(fn:(r)=> r._measurement=="{C.MEASUREMENT_RAW}" and r.run_id=="{a.run_id}" and r._field=="seq") '
                      f'|> first() |> group() |> min(column:"_time")')
    first = next(r.values["_time"] for t in first_q for r in t.records).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"[archive] watermark={wm:%Y-%m-%d %H:%M} hot-days={a.hot_days} -> archiving [{first:%Y-%m-%d}, {cutoff:%Y-%m-%d})")

    total, day = 0, first
    while day < cutoff:
        nxt = day + timedelta(days=1)
        df = q.query_data_frame(FLUX_DAY.format(bucket=cfg["bucket"], start=day.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                 stop=nxt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                 meas=C.MEASUREMENT_RAW, run=a.run_id))
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
        if len(df):
            total += write_partitions(tidy(df, a.run_id), Path(a.out), a.run_id)
        day = nxt
    print(f"[archive] wrote {total:,} rows to {a.out}")

    if a.prune and total:
        # verify: parquet row count must equal what Influx still holds for the range
        root = Path(a.out) / "measurement=energy_raw" / f"run_id={a.run_id}"
        on_disk = sum(len(pd.read_parquet(p, columns=["seq"])) for p in root.rglob("part.parquet"))
        cnt = q.query(f'from(bucket:"{cfg["bucket"]}") |> range(start: {first:%Y-%m-%dT%H:%M:%SZ}, stop: {cutoff:%Y-%m-%dT%H:%M:%SZ}) '
                      f'|> filter(fn:(r)=> r._measurement=="{C.MEASUREMENT_RAW}" and r.run_id=="{a.run_id}" and r._field=="seq") '
                      f'|> group() |> count()')
        in_influx = next((int(r.values["_value"]) for t in cnt for r in t.records), 0)
        if on_disk < in_influx:
            print(f"[archive] NOT pruning: parquet has {on_disk:,} rows but Influx has {in_influx:,}"); return 2
        cl.delete_api().delete(first, cutoff, f'_measurement="{C.MEASUREMENT_RAW}" AND run_id="{a.run_id}"',
                               bucket=cfg["bucket"], org=cfg["org"])
        print(f"[archive] pruned {in_influx:,} points from InfluxDB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
