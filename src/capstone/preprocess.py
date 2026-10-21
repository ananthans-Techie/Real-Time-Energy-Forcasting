"""Cleaning, minute-grid construction, hourly aggregation, chronological split and the
data-quality report.  This is the SINGLE implementation of the documented
missing-value / aggregation rules; the streaming aggregator (aggregator.py) is
tested for numerical parity against `aggregate_hourly` here.

Frozen rules (see docs/DATA_PREPARATION.md):
  M1  Source '?' / empty  -> NaN.  A minute is *missing* iff its target is NaN.
  M2  The minute table is reindexed to the complete 1-minute grid first..last; grid
      minutes absent from the source become missing rows too.
  M3  Hourly value = mean of the valid minutes in [h, h+1h) for every column
      (sub_metering_* therefore become mean Wh/min).
  M4  An hour with zero valid target minutes is IMPUTED by last-observation-carried-
      forward (the base paper's rule, Im et al. 2024 Sec. 4.2) and flagged imputed=True.
      A single missing feature value inside an otherwise valid hour is also LOCF'd,
      but the hour is not flagged (counted separately in the quality report).
  M5  Raw minute data is never imputed -- the stream carries is_missing=True instead.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import constants as C


# ------------------------------------------------------------------------ split
def assign_split(times) -> np.ndarray:
    """Chronological split label per timestamp (half-open cut-offs, see constants)."""
    t = pd.DatetimeIndex(times)
    if t.tz is None:
        t = t.tz_localize("UTC")
    out = np.full(len(t), "test", dtype=object)
    out[t < pd.Timestamp(C.SPLIT_VAL_END)] = "val"
    out[t < pd.Timestamp(C.SPLIT_TRAIN_END)] = "train"
    return out


# ---------------------------------------------------------------------- reading
def read_uci(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse household_power_consumption.txt (or the UCI .zip) into canonical columns.

    Returns (raw_df, info).  raw_df has event_time (UTC) + the 7 value columns,
    with '?' / blanks turned into NaN, in source order, without any gap filling.
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            member = next(n for n in z.namelist() if n.lower().endswith(".txt"))
            with z.open(member) as fh:
                raw = pd.read_csv(fh, sep=";", dtype=str, keep_default_na=False)
    else:
        raw = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)

    need = {"Date", "Time", *C.UCI_COLUMN_MAP}
    missing_cols = need - set(raw.columns)
    if missing_cols:
        raise ValueError(f"Input is missing expected UCI columns: {sorted(missing_cols)}")

    stamp = raw["Date"].str.strip() + " " + raw["Time"].str.strip()
    event_time = pd.to_datetime(stamp, format="%d/%m/%Y %H:%M:%S", utc=True)
    out = pd.DataFrame({"event_time": event_time})
    unparseable = 0
    for src, dst in C.UCI_COLUMN_MAP.items():
        s = raw[src].str.strip()
        num = pd.to_numeric(s, errors="coerce")
        unparseable += int((num.isna() & ~s.isin(["?", ""])).sum())
        out[dst] = num
    return out, {"source_rows": int(len(out)), "unparseable_values": unparseable}


def build_minute_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sort, de-duplicate and reindex to the complete 1-minute grid (rule M2)."""
    info: dict[str, Any] = {}
    info["out_of_order_rows"] = int((raw["event_time"].diff().dt.total_seconds() < 0).sum())
    df = raw.sort_values("event_time", kind="stable")
    info["duplicate_timestamps"] = int(df["event_time"].duplicated().sum())
    df = df.drop_duplicates("event_time", keep="first")

    off_grid = int((df["event_time"].dt.second != 0).sum())
    if off_grid:
        raise ValueError(f"{off_grid} timestamps are not on a whole-minute boundary; "
                         "the frozen contract assumes minute-aligned UCI rows.")

    grid = pd.date_range(df["event_time"].iloc[0], df["event_time"].iloc[-1], freq="1min")
    info["grid_minutes"] = int(len(grid))
    info["grid_minutes_absent_from_source"] = int(len(grid) - len(df))
    m = df.set_index("event_time").reindex(grid)
    m.index.name = "event_time"
    m = m.reset_index()
    m["is_missing"] = m[C.TARGET].isna()
    m["split"] = assign_split(m["event_time"])
    cols = ["event_time", *C.VALUE_COLUMNS, "is_missing", "split"]
    return m[cols], info


# ------------------------------------------------------------------- aggregation
def aggregate_hourly(minute: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the minute table to the frozen 1-hour forecast interval (rules M3, M4)."""
    m = minute.set_index("event_time")
    cols = list(C.VALUE_COLUMNS)
    hourly = m[cols].resample("1h").mean()                       # mean of valid minutes
    n_valid = m[C.TARGET].notna().resample("1h").sum().astype("int64")
    hourly["n_valid"] = n_valid
    hourly["imputed"] = hourly["n_valid"].eq(0)                  # no valid target minute
    hourly[cols] = hourly[cols].ffill()                          # LOCF
    hourly = hourly.dropna(subset=[C.TARGET])                    # leading hours, if any
    hourly.index.name = "event_time"
    hourly = hourly.reset_index()
    hourly["window_end"] = hourly["event_time"] + pd.Timedelta(seconds=C.FORECAST_INTERVAL_S)
    hourly["n_expected"] = C.MINUTES_PER_WINDOW
    hourly["split"] = assign_split(hourly["event_time"])
    return hourly[["event_time", "window_end", *C.VALUE_COLUMNS,
                   "n_valid", "n_expected", "imputed", "split"]]


# ------------------------------------------------------------- quality reporting
def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start indices and lengths of consecutive True runs."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends - starts


def quality_report(read_info: dict, grid_info: dict, minute: pd.DataFrame,
                   hourly: pd.DataFrame) -> dict[str, Any]:
    rep: dict[str, Any] = {"schema_version": C.SCHEMA_VERSION}
    rep["source"] = {
        **read_info, **grid_info,
        "expected_rows_per_paper": C.UCI_EXPECTED_ROWS,
        "row_count_matches_paper": read_info["source_rows"] == C.UCI_EXPECTED_ROWS,
        "first_event_time": minute["event_time"].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_event_time": minute["event_time"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    miss = minute["is_missing"].to_numpy()
    starts, lengths = _runs(miss)
    order = np.argsort(-lengths)[:10]
    any_feature_missing = minute[list(C.FEATURES)].isna().any(axis=1).to_numpy()
    rep["minute_missingness"] = {
        "missing_minutes": int(miss.sum()),
        "missing_pct": round(100.0 * float(miss.mean()), 4),
        "missing_runs": int(len(starts)),
        "longest_run_minutes": int(lengths.max()) if len(lengths) else 0,
        "runs_ge_60_minutes": int((lengths >= 60).sum()),
        "runs_ge_1440_minutes": int((lengths >= 1440).sum()),
        "top_runs": [
            {"start": minute["event_time"].iloc[int(starts[i])].strftime("%Y-%m-%dT%H:%M:%SZ"),
             "minutes": int(lengths[i])} for i in order
        ],
        "target_valid_but_feature_missing": int((~miss & any_feature_missing).sum()),
        "target_missing_but_feature_present": int(
            (miss & minute[list(C.FEATURES)].notna().any(axis=1).to_numpy()).sum()),
    }

    rep["value_ranges"] = {
        c: {"min": float(minute[c].min()), "max": float(minute[c].max()),
            "mean": float(minute[c].mean()),
            "negative_values": int((minute[c] < 0).sum())}
        for c in C.VALUE_COLUMNS
    }
    rep["value_ranges"]["voltage"]["zero_values"] = int((minute["voltage"] == 0).sum())

    imp = hourly["imputed"].to_numpy()
    hs, hl = _runs(imp)
    rep["hourly"] = {
        "hours": int(len(hourly)),
        "imputed_hours": int(imp.sum()),
        "imputed_pct": round(100.0 * float(imp.mean()), 4),
        "longest_imputed_run_hours": int(hl.max()) if len(hl) else 0,
        "imputed_runs_gt_24h": int((hl > 24).sum()),
        "partial_hours_excluding_edges": int(
            ((hourly["n_valid"] < C.MINUTES_PER_WINDOW) & ~hourly["imputed"]).iloc[1:-1].sum())
        if len(hourly) > 2 else 0,
    }

    splits = {}
    for s in C.SPLITS:
        mm = minute[minute["split"] == s]
        hh = hourly[hourly["split"] == s]
        if len(mm):
            splits[s] = {
                "first": mm["event_time"].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last": mm["event_time"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "minutes": int(len(mm)), "missing_pct": round(100.0 * float(mm["is_missing"].mean()), 4),
                "hours": int(len(hh)), "imputed_hours": int(hh["imputed"].sum()),
            }
        else:
            splits[s] = {"minutes": 0, "hours": 0}
    rep["splits"] = splits
    return rep


def render_quality_markdown(rep: dict[str, Any]) -> str:
    s, mm, h = rep["source"], rep["minute_missingness"], rep["hourly"]
    L = ["# UCI data-quality report", "",
         f"*Generated by `cap-prepare`; schema {rep['schema_version']}. All numbers are computed from the file you supplied.*", "",
         "## Source", "",
         f"- Rows read: **{s['source_rows']:,}** (paper: {s['expected_rows_per_paper']:,}) -> "
         f"{'matches' if s['row_count_matches_paper'] else '**DOES NOT MATCH**'}",
         f"- Span: {s['first_event_time']} -> {s['last_event_time']} (labelled UTC)",
         f"- Minute-grid length: {s['grid_minutes']:,}; grid minutes absent from source: {s['grid_minutes_absent_from_source']:,}",
         f"- Duplicate timestamps dropped: {s['duplicate_timestamps']:,}; out-of-order rows: {s['out_of_order_rows']:,}; "
         f"unparseable values: {s['unparseable_values']:,}", "",
         "## Minute-level missingness", "",
         f"- Missing minutes: **{mm['missing_minutes']:,}** ({mm['missing_pct']}%) in {mm['missing_runs']:,} runs",
         f"- Longest run: {mm['longest_run_minutes']:,} min; runs >= 60 min: {mm['runs_ge_60_minutes']}; runs >= 1 day: {mm['runs_ge_1440_minutes']}",
         f"- Target valid but a feature missing: {mm['target_valid_but_feature_missing']:,}; "
         f"target missing but a feature present: {mm['target_missing_but_feature_present']:,}", "",
         "| Longest runs (start, UTC) | Minutes |", "|---|---|"]
    L += [f"| {r['start']} | {r['minutes']:,} |" for r in mm["top_runs"]]
    L += ["", "## Hourly series (forecast interval)", "",
          f"- Hours: **{h['hours']:,}**; imputed (LOCF): **{h['imputed_hours']:,}** ({h['imputed_pct']}%)",
          f"- Longest imputed run: {h['longest_imputed_run_hours']} h; imputed runs > 24 h: {h['imputed_runs_gt_24h']}",
          f"- Partially observed hours (excluding first/last): {h['partial_hours_excluding_edges']:,}", "",
          "## Value ranges (minute level, valid values)", "",
          "| Column | min | max | mean | negatives |", "|---|---|---|---|---|"]
    for c, v in rep["value_ranges"].items():
        L.append(f"| {c} | {v['min']:.3f} | {v['max']:.3f} | {v['mean']:.3f} | {v['negative_values']} |")
    L += ["", "## Chronological split", "",
          "| Split | First | Last | Minutes | Missing % | Hours | Imputed hours |", "|---|---|---|---|---|---|---|"]
    for k, v in rep["splits"].items():
        if v["minutes"]:
            L.append(f"| {k} | {v['first']} | {v['last']} | {v['minutes']:,} | {v['missing_pct']} | {v['hours']:,} | {v['imputed_hours']:,} |")
        else:
            L.append(f"| {k} | - | - | 0 | - | 0 | 0 |")
    L += ["", "> **Reading this report.** Long imputed runs turn into flat, carried-forward segments in the "
              "hourly series. They are flagged (`imputed=True`) so every model can be scored with and "
              "without them; the contract default is to report headline MAE/RMSE on `imputed == False` hours."]
    return "\n".join(L) + "\n"
