#!/usr/bin/env python3
"""Run full Workstream 3 benchmark evaluation and generate metrics report.

Usage:
  python scripts/evaluate_workstream3.py --data data/processed/uci_hourly.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from cap_anomaly import AnomalyDetector, AnomalyEvaluator, SyntheticFaultInjector
from cap_drift import DriftMonitor
from capstone import constants as C


def run_benchmark(data_path: str, out_report_path: str = "runs/workstream3_evaluation_report.md") -> int:
    path = Path(data_path)
    if not path.exists():
        print(f"Dataset {path} not found. Synthesizing test dataset...")
        from capstone.synthetic import make_synthetic, to_uci_text
        from capstone.preprocess import read_uci, build_minute_table, aggregate_hourly
        import tempfile
        df_raw = make_synthetic("2009-12-25 00:00:00", 16, seed=12)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write(to_uci_text(df_raw))
            tmp_path = Path(tmp.name)
        try:
            raw, _ = read_uci(tmp_path)
            minute_df, _ = build_minute_table(raw)
            df = aggregate_hourly(minute_df)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        df = pd.read_parquet(path)

    # 1. Inject benchmark fault scenarios
    injector = SyntheticFaultInjector(seed=42)
    injected_df, labels_df = injector.inject_benchmark_suite(df)

    # 2. Run Anomaly Detection over the injected stream
    detector = AnomalyDetector(window_size=24, threshold=3.5)
    detected_anomalies = []
    for _, row in injected_df.iterrows():
        t_iso = row["event_time"].strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(row["event_time"], "strftime") else str(row["event_time"])
        rec = {
            "run_id": "WS3_benchmark-r1",
            "event_id": f"uci-001-{t_iso.replace('-', '').replace(':', '')}",
            "meter_id": "uci-001",
            "event_time": t_iso,
            "target": row.get("target"),
            "is_missing": bool(row.get("is_missing", False)),
            "trigger_replay_time_ms": 1792500000000,
        }
        ev = detector.process_record(rec)
        if ev is not None:
            detected_anomalies.append(ev)

    anomaly_metrics = AnomalyEvaluator.evaluate_anomalies(labels_df, detected_anomalies)

    # 3. Simulate drift monitoring over residuals
    drift_monitor = DriftMonitor(method="adwin", delta=0.005)
    drift_events = []
    # Baseline normal residuals ~ 0.15 kW, during level_shift jumps to ~ 1.75 kW
    for _, row in injected_df.iterrows():
        t_iso = row["event_time"].strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(row["event_time"], "strftime") else str(row["event_time"])
        target_val = row.get("target")
        if target_val is not None and not pd.isna(target_val):
            # Model forecast baseline expectation
            pred_baseline = 1.2
            residual = abs(float(target_val) - pred_baseline)
            dev = drift_monitor.process_error(
                run_id="WS3_benchmark-r1",
                meter_id="uci-001",
                event_time_iso=t_iso,
                error_val=residual,
                trigger_event_id=f"uci-001-{t_iso.replace('-', '').replace(':', '')}",
                trigger_replay_time_ms=1792500000000,
            )
            drift_events.append(dev)

    drift_metrics = AnomalyEvaluator.evaluate_drift(
        ground_truth_labels=labels_df,
        detected_drifts=drift_events,
        stream_event_times=injected_df["event_time"].to_list(),
    )

    report_md = AnomalyEvaluator.generate_markdown_report(anomaly_metrics, drift_metrics)

    out_p = Path(out_report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(report_md)

    print(f"\nReport generated -> {out_p}\n")
    print(report_md)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/uci_hourly.parquet")
    ap.add_argument("--report", default="runs/workstream3_evaluation_report.md")
    a = ap.parse_args(argv)
    return run_benchmark(a.data, a.report)


if __name__ == "__main__":
    sys.exit(main())
