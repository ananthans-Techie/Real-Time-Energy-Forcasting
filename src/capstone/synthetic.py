"""Generate a synthetic file in the exact UCI text format (semicolon separated,
non-zero-padded d/m/yyyy dates, '?' for missing).  For smoke tests and CI only --
NOT for any reported result.

  cap-synth --start 2009-09-28 --days 110 --out data/raw/synthetic_uci.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HEADER = ("Date;Time;Global_active_power;Global_reactive_power;Voltage;"
          "Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3")


def make_synthetic(start: str, days: int, seed: int = 7, long_gap_day: int | None = 10,
                   missing_runs: int = 12) -> pd.DataFrame:
    """Return a DataFrame with the UCI column names; NaN marks missing minutes."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=days * 1440, freq="1min")
    hod = (t.hour + t.minute / 60).to_numpy()
    dow = t.dayofweek.to_numpy()
    base = 0.6 + 1.2 * np.exp(-((hod - 19) ** 2) / 6) + 0.6 * np.exp(-((hod - 7.5) ** 2) / 3)
    base += 0.3 * (dow >= 5) + 0.25 * np.cos(2 * np.pi * t.dayofyear.to_numpy() / 365)
    gap = np.clip(base + rng.gamma(2.0, 0.15, len(t)) - 0.3, 0.08, None)
    spikes = rng.random(len(t)) < 0.004
    gap = gap + spikes * rng.uniform(2, 4, len(t))
    volt = 240 + rng.normal(0, 1.5, len(t)) - 1.2 * gap
    df = pd.DataFrame({
        "Global_active_power": gap.round(3),
        "Global_reactive_power": (0.1 + 0.12 * gap + rng.normal(0, 0.02, len(t))).clip(0).round(3),
        "Voltage": volt.round(3),
        "Global_intensity": (gap * 1000 / volt).round(1),
        "Sub_metering_1": (rng.random(len(t)) < 0.03) * rng.integers(1, 30, len(t)),
        "Sub_metering_2": (rng.random(len(t)) < 0.05) * rng.integers(1, 40, len(t)),
        "Sub_metering_3": (rng.random(len(t)) < 0.4) * rng.integers(1, 20, len(t)),
    }, index=t).astype(float)
    # missing blocks (all columns at once, like the real file)
    for _ in range(missing_runs):
        s = int(rng.integers(0, len(t) - 400)); n = int(rng.integers(1, 90))
        df.iloc[s:s + n] = np.nan
    if long_gap_day is not None and days > long_gap_day + 4:
        s = long_gap_day * 1440 + 300
        df.iloc[s:s + 3 * 1440] = np.nan                       # a 3-day outage
    df.index.name = "ts"
    return df


def to_uci_text(df: pd.DataFrame) -> str:
    lines = [HEADER]
    for ts, row in df.iterrows():
        d = f"{ts.day}/{ts.month}/{ts.year}"
        vals = ["?" if pd.isna(v) else (f"{v:.3f}" if i < 3 else f"{v:.1f}" if i == 3 else f"{v:.1f}")
                for i, v in enumerate(row.to_numpy())]
        lines.append(f"{d};{ts:%H:%M:%S};" + ";".join(vals))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2009-09-28")
    ap.add_argument("--days", type=int, default=110)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/raw/synthetic_uci.txt")
    a = ap.parse_args(argv)
    df = make_synthetic(a.start, a.days, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_uci_text(df))
    print(f"wrote {len(df):,} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
