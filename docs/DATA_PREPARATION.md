# Data preparation — cleaning, missing-value handling, aggregation, split

Implemented once in `src/capstone/preprocess.py`; the streaming aggregator (`aggregator.py`) is unit-tested for **numerical parity** with it (`tests/test_aggregator.py::test_stream_matches_batch`), so offline training data and the live stream cannot drift apart.

```
cap-prepare --download            # or: --input data/raw/household_power_consumption.zip
```

## Pipeline

1. **Parse.** Semicolon-separated UCI text (or the `.zip`). `Date` is `d/m/yyyy` (often not zero-padded), `Time` is `HH:MM:SS`. `?` and blanks → NaN. Anything non-numeric that is *not* `?` is counted as `unparseable_values` in the report (expected 0).
2. **Minute grid.** Sort, drop duplicate timestamps (counted), and reindex to the complete 1-minute grid first→last. The UCI file should already be complete (2,075,259 rows = every minute between 2006-12-16 17:24 and 2010-11-26 21:02); the report states whether the count matches the base paper and how many grid minutes were absent. Timestamps off a whole minute abort the run rather than being silently dropped.
3. **Flag, don't fill, at minute level.** `is_missing = target is NaN`. Raw minutes are never imputed. This is deliberate: the anomaly detector's "missing-data cases" and the event-time processor's "documented missing-value rules" both need to *see* the gap.
4. **Aggregate to the frozen 1-hour interval** (mean of valid minutes per column).
5. **Impute at hour level (LOCF).** Hours with zero valid target minutes copy the previous hour's values and get `imputed=True`.
6. **Split chronologically** (`train` < 2009-10-01 ≤ `val` < 2010-01-01 ≤ `test`).
7. **Write** `uci_minute.parquet`, `uci_hourly.parquet`, `quality_report.{md,json}`, `dataset_manifest.json` (content SHA-256 — all three of us compare this hash before running anything).

## Missing-value rules (frozen)

| Rule | Statement | Rationale |
|---|---|---|
| M1 | `?`/blank → NaN; a minute is *missing* iff its target is NaN | UCI missing rows are understood to blank all measurements together; the report counts any partial cases (expected 0) rather than assuming |
| M2 | Reindex to the complete minute grid; absent grid minutes are missing too | A stream needs an explicit event per interval to reason about gaps |
| M3 | Hourly value = mean of valid minutes (`sub_metering_*` → mean Wh/min) | Unbiased by the number of missing minutes; keeps `target` in kW |
| M4 | Hour with 0 valid minutes → LOCF + `imputed=True`. A single missing *feature* inside an otherwise valid hour is also carried forward but not flagged (counted in the report) | The base paper's stated rule (Im et al. 2024 §4.2: "replace missing values with the most recent value"), made auditable |
| M5 | Streams carry `is_missing=true` with value keys **absent**; no minute-level imputation anywhere | Keeps raw data honest |

## Caveats to state in the report

* **LOCF produces flat segments.** A long outage becomes a constant line in the hourly series; a forecaster can look artificially good (it learns "repeat the last value") or bad. That is why `imputed` exists: report headline MAE/RMSE on `imputed == False` hours and all-hours side by side. Read the quality report's *longest imputed run* and *runs > 24 h* lines before choosing a policy; Figure 2 of the base paper shows a conspicuous low stretch in mid-2008, so check whether it appears there as missing data or is genuine low consumption.
* **Partial edge hours.** The first and last hours are partial (the file starts at 17:24 and ends at 21:02); they are aggregated from the minutes that exist.
* **Timezone.** UCI gives no timezone. Stamps are labelled UTC as-is, so the 24 h season is exact. If the household's clock followed DST, the seasonal pattern will shift by an hour twice a year in clock terms — a limitation, not a bug.
* **Sub-metering units.** Wh of active energy per minute, so hourly means are Wh/min; multiply by 60/1000 for kWh-per-hour-equivalent power. `1000·target/60 − Σ sub_metering` is the UCI "unmetered" energy; it is not part of the frozen feature set.
* **No labels.** UCI has no anomaly/drift labels; evaluation uses injected, documented ground truth (abstract §8).
* **Base-paper comparability.** Same dataset, same hourly interval, SARIMA (1,0,1,24), training through Dec 2009, evaluation on 2010. The base paper's Table 1 (SARIMA MAE 0.3592 / RMSE 0.5103; LSTM MAE 0.1084 / RMSE 0.0825) is the sanity check for Aleena's fixed baselines — large departures usually mean a different split or imputation, not a different model.

## Reproducing

```bash
cap-prepare --download
cat data/processed/dataset_manifest.json       # compare content_sha256 across the team
less data/processed/quality_report.md
```

No real-data numbers are quoted in this repository: they come from the report generated on your copy of the file.
