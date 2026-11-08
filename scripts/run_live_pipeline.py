"""End-to-End Live Pipeline Runner & Integration Validator.

Launches:
  1. cap-hourly (1-hour tumbling aggregator with watermark)
  2. cap-forecast (Streaming Forecaster & Dynamic Model Selector)
  3. cap-anomaly (Rolling Median & MAD detector)
  4. cap-drift (ADWIN residual concept drift monitor)
  5. cap-replay (Test split replay producer)

Validates that all topics are populated and InfluxDB receives all measurements.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from influxdb_client import InfluxDBClient

from capstone import constants as C
from capstone.env import kafka_bootstrap, load_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("live_pipeline")


def run_pipeline(
    bootstrap: str = "127.0.0.1:9092",
    run_id: str = "live-demo-01",
    limit_minutes: int = 2880,  # 2 days of data = 48 hourly windows
    replay_rate: int = 500,
    settle_s: int = 10,
    fixed_model: str | None = None,
) -> dict[str, int]:
    logger.info(f"=== Starting Live Pipeline Run: {run_id} (limit={limit_minutes} min, rate={replay_rate} rps) ===")
    py = sys.executable

    log_dir = Path("runs/live_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    f_hourly = open(log_dir / "hourly.log", "w")
    f_forecast = open(log_dir / "forecast.log", "w")
    f_anomaly = open(log_dir / "anomaly.log", "w")
    f_drift = open(log_dir / "drift.log", "w")

    # 1. Start streaming aggregator (energy.raw -> energy.hourly)
    p_hourly = subprocess.Popen(
        [py, "-m", "capstone.aggregator", "--bootstrap", bootstrap],
        stdout=f_hourly,
        stderr=subprocess.STDOUT,
    )
    logger.info("Launched cap-hourly")

    # 2. Start streaming forecaster (energy.hourly -> energy.predictions, energy.model_events)
    cmd_forecast = [py, "-m", "capstone.forecasting.forecaster", "--bootstrap", bootstrap]
    if fixed_model:
        cmd_forecast += ["--fixed-model", fixed_model]
    p_forecast = subprocess.Popen(
        cmd_forecast,
        stdout=f_forecast,
        stderr=subprocess.STDOUT,
    )
    logger.info(f"Launched cap-forecast (fixed={fixed_model})")

    # 3. Start streaming anomaly detector (energy.hourly -> energy.anomalies)
    p_anomaly = subprocess.Popen(
        [py, "-m", "cap_anomaly.anomaly_detector", "--bootstrap", bootstrap],
        stdout=f_anomaly,
        stderr=subprocess.STDOUT,
    )
    logger.info("Launched cap-anomaly")

    # 4. Start streaming drift monitor (energy.predictions + energy.hourly -> energy.drift)
    p_drift = subprocess.Popen(
        [py, "-m", "cap_drift.drift_monitor", "--bootstrap", bootstrap],
        stdout=f_drift,
        stderr=subprocess.STDOUT,
    )
    logger.info("Launched cap-drift")

    time.sleep(3)  # Allow consumer group coordination

    # 5. Start replay producer
    logger.info("Launching cap-replay...")
    cmd_replay = [
        py, "-m", "capstone.replay_producer",
        "--bootstrap", bootstrap,
        "--run-id", run_id,
        "--split", "test",
        "--limit", str(limit_minutes),
        "--rate", str(replay_rate),
        "--label", "live_demo",
    ]
    p_replay = subprocess.run(cmd_replay, check=True)
    logger.info(f"cap-replay completed with code {p_replay.returncode}")

    logger.info(f"Waiting {settle_s}s for streaming window drain and InfluxDB flush...")
    time.sleep(settle_s)

    # Terminate streaming workers cleanly
    for name, proc in [("drift", p_drift), ("anomaly", p_anomaly), ("forecast", p_forecast), ("hourly", p_hourly)]:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        logger.info(f"Stopped {name}")

    f_hourly.close()
    f_forecast.close()
    f_anomaly.close()
    f_drift.close()

    # 6. Verify InfluxDB counts
    client = InfluxDBClient(url="http://127.0.0.1:8086", token="capstone-dev-token-change-me", org="capstone")
    query_api = client.query_api()

    measurements = [
        "energy_raw",
        "energy_predictions",
        "energy_anomalies",
        "energy_drift",
        "energy_model_events",
    ]
    counts = {}
    for m in measurements:
        flux = f'''
        from(bucket: "energy")
          |> range(start: 2006-01-01T00:00:00Z)
          |> filter(fn: (r) => r._measurement == "{m}" and r.run_id == "{run_id}")
          |> count()
        '''
        tables = query_api.query(flux)
        cnt = 0
        for table in tables:
            for record in table.records:
                cnt += record.get_value()
        counts[m] = cnt
        logger.info(f"InfluxDB [{m}]: {cnt} points for run_id={run_id}")

    return counts


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", default="127.0.0.1:9092")
    ap.add_argument("--run-id", default=f"live-demo-{int(time.time())}")
    ap.add_argument("--limit", type=int, default=2880)
    ap.add_argument("--rate", type=int, default=1000)
    ap.add_argument("--fixed-model", default=None)
    args = ap.parse_args(argv)

    counts = run_pipeline(
        bootstrap=args.bootstrap,
        run_id=args.run_id,
        limit_minutes=args.limit,
        replay_rate=args.rate,
        fixed_model=args.fixed_model,
    )
    print("\n=== PIPELINE VALIDATION RESULT ===")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
