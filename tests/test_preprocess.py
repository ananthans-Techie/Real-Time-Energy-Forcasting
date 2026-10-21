import numpy as np
import pandas as pd

from capstone import constants as C
from capstone.preprocess import assign_split, aggregate_hourly, read_uci, build_minute_table, quality_report


def test_split_cutoffs_are_half_open():
    t = pd.to_datetime(["2009-09-30T23:59:00Z", "2009-10-01T00:00:00Z",
                        "2009-12-31T23:59:00Z", "2010-01-01T00:00:00Z"], utc=True)
    assert list(assign_split(t)) == ["train", "val", "val", "test"]


def test_parser_handles_unpadded_dates_and_question_marks(uci_file):
    raw, info = read_uci(uci_file)
    assert info["unparseable_values"] == 0
    assert raw["event_time"].dt.tz is not None
    assert raw[C.TARGET].isna().sum() > 0


def test_minute_grid_is_complete_and_flags_missing(minute):
    d = minute["event_time"].diff().dropna().dt.total_seconds()
    assert (d == 60).all()
    assert (minute["is_missing"] == minute[C.TARGET].isna()).all()
    assert set(minute["split"]) == {"val", "test"}


def test_hourly_rules(minute, hourly):
    # M3: mean over valid minutes
    h0 = hourly.iloc[5]
    w = minute[(minute["event_time"] >= h0["event_time"]) & (minute["event_time"] < h0["window_end"])]
    if h0["n_valid"] > 0:
        assert np.isclose(h0["target"], w[C.TARGET].mean())
    # n_valid never exceeds the window; first hour is partial because the file starts at 00:07
    assert hourly["n_valid"].max() <= 60 and hourly.iloc[0]["n_valid"] <= 53
    # M4: imputed <=> no valid minute, and the value equals the previous hour (LOCF)
    imp = hourly[hourly["imputed"]]
    assert len(imp) >= 4                                         # the 5-hour outage
    assert (imp["n_valid"] == 0).all()
    for i in imp.index:
        assert hourly.loc[i, C.TARGET] == hourly.loc[i - 1, C.TARGET]
    # complete hourly grid, no NaN left
    assert (hourly["event_time"].diff().dropna().dt.total_seconds() == 3600).all()
    assert not hourly[list(C.VALUE_COLUMNS)].isna().any().any()


def test_quality_report_numbers(uci_file, minute, hourly):
    raw, ri = read_uci(uci_file)
    _, gi = build_minute_table(raw)
    q = quality_report(ri, gi, minute, hourly)
    assert q["minute_missingness"]["missing_minutes"] == int(minute["is_missing"].sum())
    assert q["hourly"]["imputed_hours"] == int(hourly["imputed"].sum())
    assert q["minute_missingness"]["longest_run_minutes"] >= 300
