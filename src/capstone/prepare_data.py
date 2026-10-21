"""cap-prepare: clean the UCI file and write the frozen, shared datasets.

  data/processed/uci_minute.parquet      1-min table, complete grid, is_missing + split
  data/processed/uci_hourly.parquet      1-h table (frozen forecast interval), LOCF + flags
  data/processed/quality_report.{json,md}
  data/processed/dataset_manifest.json   row counts + content hashes: everyone verifies the same data

Usage:
  cap-prepare --input data/raw/household_power_consumption.zip
  cap-prepare --download            # fetch from the UCI repository first
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from . import constants as C
from .preprocess import (aggregate_hourly, build_minute_table, quality_report,
                         read_uci, render_quality_markdown)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"


def content_hash(df: pd.DataFrame) -> str:
    """Hash of the data values (independent of parquet writer version)."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    h.update(",".join(df.columns).encode())
    return h.hexdigest()


def prepare(input_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, read_info = read_uci(input_path)
    minute, grid_info = build_minute_table(raw)
    hourly = aggregate_hourly(minute)

    minute_path, hourly_path = out_dir / "uci_minute.parquet", out_dir / "uci_hourly.parquet"
    minute.to_parquet(minute_path, engine="pyarrow", compression="zstd", index=False)
    hourly.to_parquet(hourly_path, engine="pyarrow", compression="zstd", index=False)

    rep = quality_report(read_info, grid_info, minute, hourly)
    (out_dir / "quality_report.json").write_text(json.dumps(rep, indent=2))
    (out_dir / "quality_report.md").write_text(render_quality_markdown(rep))

    manifest = {
        "schema_version": C.SCHEMA_VERSION,
        "source": "UCI Individual Household Electric Power Consumption (Hebrail & Berard, 2012)",
        "forecast_interval": C.FORECAST_INTERVAL_ISO,
        "forecast_horizon_steps": C.FORECAST_HORIZON_STEPS,
        "seasonal_period": C.SEASONAL_PERIOD,
        "timezone_note": C.TIMEZONE_NOTE,
        "split_cutoffs": {"train_end_exclusive": C.SPLIT_TRAIN_END, "val_end_exclusive": C.SPLIT_VAL_END},
        "files": {
            "uci_minute.parquet": {"rows": int(len(minute)), "content_sha256": content_hash(minute)},
            "uci_hourly.parquet": {"rows": int(len(hourly)), "content_sha256": content_hash(hourly)},
        },
        "splits_hourly": {s: int((hourly["split"] == s).sum()) for s in C.SPLITS},
        "splits_minute": {s: int((minute["split"] == s).sum()) for s in C.SPLITS},
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"quality": rep, "manifest": manifest}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/raw/household_power_consumption.zip",
                    help=".zip from UCI or the extracted household_power_consumption.txt")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--download", action="store_true", help="download the UCI zip to --input first")
    a = ap.parse_args(argv)

    inp = Path(a.input)
    if a.download:
        inp.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {UCI_ZIP_URL} -> {inp}")
        try:
            urllib.request.urlretrieve(UCI_ZIP_URL, inp)
        except Exception as e:  # noqa: BLE001
            print(f"Download failed ({e}). Download the zip manually from "
                  "https://archive.ics.uci.edu/dataset/235 and pass it via --input.", file=sys.stderr)
            return 2
    if not inp.exists():
        print(f"Input not found: {inp}. Use --download or place the UCI file there.", file=sys.stderr)
        return 2

    res = prepare(inp, Path(a.out))
    q = res["quality"]
    print(f"minute rows : {res['manifest']['files']['uci_minute.parquet']['rows']:,}")
    print(f"hourly rows : {res['manifest']['files']['uci_hourly.parquet']['rows']:,}")
    print(f"missing min : {q['minute_missingness']['missing_minutes']:,} ({q['minute_missingness']['missing_pct']}%)")
    print(f"imputed hrs : {q['hourly']['imputed_hours']:,} ({q['hourly']['imputed_pct']}%)")
    print(f"row count matches paper: {q['source']['row_count_matches_paper']}")
    print(f"reports in {a.out}/quality_report.md and dataset_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
