"""cap-probe: measure end-to-end latency (producer emit -> record queryable in InfluxDB).

Method: poll InfluxDB for the newest stored reading of the primary meter. Each time the visible
`seq` advances, record  latency_ms = (time the query returned) - replay_time_ms  of that reading.
The true latency lies within [latency - poll_interval - query_time, latency]; keep
--poll-ms small (default 200) and report the resolution next to the numbers.

Also reports drain_time_s: producer end -> final record visible.

  cap-probe --manifest runs/A_ramp/manifest.json --poll-ms 200
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from . import constants as C
from .env import influx_settings, kafka_bootstrap, load_env
from .metrics import MetricsPublisher

FLUX_LAST = '''
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r._measurement == "{meas}" and r.run_id == "{run}" and r.meter_id == "{meter}")
  |> filter(fn: (r) => r._field == "seq" or r._field == "replay_time_ms")
  |> last()
  |> group(columns: ["run_id", "meter_id"])
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
'''


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(latencies_ms: list[float]) -> dict:
    return {"samples": len(latencies_ms),
            "p50_ms": percentile(latencies_ms, 50), "p95_ms": percentile(latencies_ms, 95),
            "p99_ms": percentile(latencies_ms, 99), "max_ms": max(latencies_ms, default=float("nan")),
            "mean_ms": statistics.fmean(latencies_ms) if latencies_ms else float("nan")}


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--poll-ms", type=int, default=200)
    ap.add_argument("--timeout-s", type=float, default=3600)
    ap.add_argument("--bootstrap", default=kafka_bootstrap())
    ap.add_argument("--no-publish", action="store_true", help="do not emit e2e.latency_ms to energy.metrics")
    a = ap.parse_args(argv)

    from influxdb_client import InfluxDBClient
    man = json.loads(Path(a.manifest).read_text())
    run, meter = man["run_id"], man["meters"][0]
    final_seq = (man["source"]["minutes"] - 1) * len(man["meters"])
    cfg = influx_settings()
    client = InfluxDBClient(url=cfg["url"], token=cfg["token"], org=cfg["org"], timeout=10_000)
    q = client.query_api()

    pub = None
    if not a.no_publish:
        from confluent_kafka import Producer
        pub = MetricsPublisher(Producer({"bootstrap.servers": a.bootstrap, "linger.ms": 20}))

    start, stop = man["source"]["start"], "2100-01-01T00:00:00Z"
    out_dir = Path(a.manifest).parent
    last_seq, samples, t_final = -1, [], None
    t_begin = time.time()
    with open(out_dir / "latency.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["observed_ms", "seq", "replay_time_ms", "latency_ms"])
        print(f"[probe] run={run} meter={meter}; waiting for data (poll {a.poll_ms} ms)")
        while time.time() - t_begin < a.timeout_s:
            t_poll = time.time()
            try:
                tables = q.query(FLUX_LAST.format(bucket=cfg["bucket"], start=start, stop=stop,
                                                  meas=C.MEASUREMENT_RAW, run=run, meter=meter))
            except Exception as e:  # noqa: BLE001
                print("[probe] query error:", e); time.sleep(1); continue
            obs_ms = int(time.time() * 1000)
            rec = next((r for t in tables for r in t.records), None)
            if rec is not None:
                seq = int(rec.values["seq"]); rt = int(rec.values["replay_time_ms"])
                if seq > last_seq:
                    lat = obs_ms - rt
                    samples.append(lat); last_seq = seq
                    w.writerow([obs_ms, seq, rt, lat]); fh.flush()
                    if pub:
                        pub.publish(run, "probe", "e2e.latency_ms", lat, "raw->influx", obs_ms)
                    start = rec.values["_time"].strftime("%Y-%m-%dT%H:%M:%SZ")   # narrow future scans
                if seq >= final_seq:
                    t_final = obs_ms
                    break
            time.sleep(max(0.0, a.poll_ms / 1000 - (time.time() - t_poll)))

    s = summarize(samples)
    s.update({"run_id": run, "poll_ms": a.poll_ms, "final_record_visible": t_final is not None})
    prod_sum = out_dir / "producer_summary.json"
    if t_final and prod_sum.exists():
        s["drain_time_s"] = round((t_final - json.loads(prod_sum.read_text())["last_replay_time_ms"]) / 1000, 3)
    (out_dir / "latency_summary.json").write_text(json.dumps(s, indent=2))
    print("[probe]", json.dumps(s))
    if pub:
        pub.producer.flush(5)
    return 0 if t_final else 1


if __name__ == "__main__":
    raise SystemExit(main())
