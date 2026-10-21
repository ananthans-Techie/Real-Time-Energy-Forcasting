import numpy as np
import pandas as pd
import pytest

from capstone.preprocess import aggregate_hourly, build_minute_table, read_uci
from capstone.synthetic import make_synthetic, to_uci_text


@pytest.fixture(scope="session")
def uci_file(tmp_path_factory):
    """~6 days straddling the val/test boundary, with missing runs and a 5-hour outage."""
    df = make_synthetic("2009-12-29 00:07:00", 6, seed=3, long_gap_day=None, missing_runs=10)
    df.iloc[2000:2300] = np.nan                      # 5 h outage -> imputed hours
    p = tmp_path_factory.mktemp("uci") / "household_power_consumption.txt"
    p.write_text(to_uci_text(df))
    return p


@pytest.fixture(scope="session")
def minute(uci_file):
    raw, _ = read_uci(uci_file)
    m, _ = build_minute_table(raw)
    return m


@pytest.fixture(scope="session")
def hourly(minute):
    return aggregate_hourly(minute)
