"""Synthetic Fault Injector for Anomaly Detection and Concept Drift Benchmarks.

Generates reproducible, labeled test scenarios strictly in accordance with
SCHEMA_CONTRACT_v1.0.0.md §6:
  - Injected data file preserves the exact schema and continuous time grid of uci_minute / uci_hourly.
  - Modifies only the 'test' split (event_time >= 2010-01-01T00:00:00Z).
  - Ground-truth labels are written to an isolated sidecar 'labels.parquet' (and never travel over Kafka).
  - Supported labeled events:
      * 'spike': Isolated positive point load surge (+2.0 kW to +5.0 kW)
      * 'drop': Isolated downward plunge (-1.0 kW to -3.0 kW)
      * 'dropout': Zero-reading sensor failure (0.0 kW reading while system active)
      * 'missing': Sensor dropout / communications blackout (is_missing=True, values dropped)
      * 'level_shift': Structural regime change (sustained step increase/decrease in mean load)
      * 'pattern_shift' / 'variance_shift': Multiplied volatility and oscillation amplitude
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from capstone import constants as C


class SyntheticFaultInjector:
    """Injects calibrated anomaly and concept drift events into telemetry datasets."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def inject_benchmark_suite(
        self,
        df: pd.DataFrame,
        target_col: str = C.TARGET,
        time_col: str = "event_time",
        split_col: str = "split",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply a comprehensive suite of labeled faults to the test split.

        Returns:
            (injected_df, labels_df)
        """
        injected = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(injected[time_col]):
            injected[time_col] = pd.to_datetime(injected[time_col], utc=True)

        # Restrict candidate injection rows to 'test' split only
        if split_col in injected.columns:
            test_mask = injected[split_col] == "test"
        else:
            test_mask = injected[time_col] >= pd.Timestamp(C.SPLIT_VAL_END)

        test_indices = injected.index[test_mask].to_numpy()
        if len(test_indices) == 0:
            raise ValueError("No rows found in the 'test' split (>= 2010-01-01T00:00:00Z).")

        labels: list[dict[str, Any]] = []
        n_test = len(test_indices)

        # -------------------------------------------------------------
        # 1. Point Spikes (Isolated high load surges)
        # -------------------------------------------------------------
        spike_offsets = [int(n_test * 0.05), int(n_test * 0.15), int(n_test * 0.28)]
        for seg_id, off in enumerate(spike_offsets, start=1):
            if off < n_test:
                idx = test_indices[off]
                orig_val = float(injected.at[idx, target_col]) if not pd.isna(injected.at[idx, target_col]) else 1.0
                mag = float(self.rng.uniform(3.5, 6.0))
                injected.at[idx, target_col] = orig_val + mag
                if "is_missing" in injected.columns:
                    injected.at[idx, "is_missing"] = False
                labels.append({
                    "event_time": injected.at[idx, time_col],
                    "label": "spike",
                    "magnitude": mag,
                    "segment_id": f"spike_{seg_id}",
                    "duration_steps": 1,
                })

        # -------------------------------------------------------------
        # 2. Zero Dropouts (Sudden 0.0 kW without missing flag)
        # -------------------------------------------------------------
        drop_offsets = [int(n_test * 0.10), int(n_test * 0.22)]
        for seg_id, off in enumerate(drop_offsets, start=1):
            if off < n_test:
                idx = test_indices[off]
                injected.at[idx, target_col] = 0.0
                if "is_missing" in injected.columns:
                    injected.at[idx, "is_missing"] = False
                labels.append({
                    "event_time": injected.at[idx, time_col],
                    "label": "dropout",
                    "magnitude": 0.0,
                    "segment_id": f"dropout_{seg_id}",
                    "duration_steps": 1,
                })

        # -------------------------------------------------------------
        # 3. Missing Data Gaps (Communication blackout)
        # -------------------------------------------------------------
        gap_start_off = int(n_test * 0.35)
        gap_len = min(60, max(5, n_test // 100))  # e.g. 60 minutes or 5 hours
        if gap_start_off + gap_len < n_test:
            gap_indices = test_indices[gap_start_off : gap_start_off + gap_len]
            for idx in gap_indices:
                injected.at[idx, target_col] = np.nan
                if "is_missing" in injected.columns:
                    injected.at[idx, "is_missing"] = True
                labels.append({
                    "event_time": injected.at[idx, time_col],
                    "label": "missing",
                    "magnitude": np.nan,
                    "segment_id": "missing_gap_1",
                    "duration_steps": gap_len,
                })

        # -------------------------------------------------------------
        # 4. Sustained Level Shift (Concept Drift: Mean load increases)
        # -------------------------------------------------------------
        shift_start_off = int(n_test * 0.50)
        shift_len = min(500, n_test // 4)  # sustained over hundreds of steps
        if shift_start_off + shift_len < n_test:
            shift_indices = test_indices[shift_start_off : shift_start_off + shift_len]
            shift_mag = 1.75  # +1.75 kW sustained level shift
            for idx in shift_indices:
                cur = injected.at[idx, target_col]
                if not pd.isna(cur):
                    injected.at[idx, target_col] = float(cur) + shift_mag
                labels.append({
                    "event_time": injected.at[idx, time_col],
                    "label": "level_shift",
                    "magnitude": shift_mag,
                    "segment_id": "regime_level_shift_1",
                    "duration_steps": shift_len,
                })

        # -------------------------------------------------------------
        # 5. Variance / Pattern Shift (Concept Drift: 3.5x volatility)
        # -------------------------------------------------------------
        var_start_off = int(n_test * 0.80)
        var_len = min(400, n_test // 5)
        if var_start_off + var_len < n_test:
            var_indices = test_indices[var_start_off : var_start_off + var_len]
            mean_baseline = float(injected.loc[var_indices, target_col].mean())
            for idx in var_indices:
                cur = injected.at[idx, target_col]
                if not pd.isna(cur):
                    # Amplify distance from mean by 3.5x
                    diff = float(cur) - mean_baseline
                    injected.at[idx, target_col] = max(0.05, mean_baseline + diff * 3.5)
                labels.append({
                    "event_time": injected.at[idx, time_col],
                    "label": "pattern_shift",
                    "magnitude": 3.5,
                    "segment_id": "regime_variance_shift_1",
                    "duration_steps": var_len,
                })

        labels_df = pd.DataFrame(labels)
        if not labels_df.empty:
            labels_df["event_time"] = pd.to_datetime(labels_df["event_time"], utc=True)
            labels_df = labels_df.sort_values("event_time").reset_index(drop=True)

        return injected, labels_df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/processed/uci_minute.parquet")
    ap.add_argument("--out-data", default="data/processed/injected_minute.parquet")
    ap.add_argument("--out-labels", default="data/processed/labels.parquet")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    src_p = Path(a.source)
    if not src_p.exists():
        print(f"Error: Source file {src_p} does not exist.")
        return 1

    df = pd.read_parquet(src_p)
    injector = SyntheticFaultInjector(seed=a.seed)
    injected_df, labels_df = injector.inject_benchmark_suite(df)

    Path(a.out_data).parent.mkdir(parents=True, exist_ok=True)
    injected_df.to_parquet(a.out_data)
    labels_df.to_parquet(a.out_labels)

    print(f"Successfully generated injected benchmark suite:")
    print(f"  Data:   {a.out_data} ({len(injected_df):,} rows)")
    print(f"  Labels: {a.out_labels} ({len(labels_df):,} labeled timestamps)")
    print(f"  Fault Breakdown:\n{labels_df['label'].value_counts().to_string()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
