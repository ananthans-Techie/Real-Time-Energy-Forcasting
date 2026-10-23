# Shared Contract v1.0.0 — event schema, prediction interval, splits, identifiers

**Status:** PROPOSED → becomes FROZEN when all three owners tick the sign-off box in §9.
**Owner:** Ananthan S (data / platform / integration). **Reviewers:** Aleena Antony (forecasting / selection), Angel Treesa John (anomaly / drift).
**Machine-readable source of truth:** `src/capstone/constants.py` and `src/capstone/schemas/*.json`. If this document and those files ever disagree, the files win and this document has a bug.

> Handoff contract #1 (architecture page 2): *"You publish and freeze the timestamped input schema and prediction interval before parallel work begins."* This is that publication.

---

## 1. Frozen decisions at a glance

| # | Decision | Frozen value | Reason / source |
|---|---|---|---|
| 1 | Source data | UCI Individual Household Electric Power Consumption, 1-min, 2,075,259 rows, Dec 2006 – Nov 2010 | Base paper §5.2; abstract §8 |
| 2 | Target | `target` = `Global_active_power`, **kW** | Base paper |
| 3 | Input features | `global_reactive_power` (kW), `voltage` (V), `global_intensity` (A), `sub_metering_1/2/3` (Wh per minute) | The six UCI independent variables |
| 4 | Event time | `event_time`, ISO-8601 UTC with `Z`, minute resolution. UCI wall-clock stamps are **labelled UTC as-is** (no DST shift) | Keeps hourly windows and the 24 h season exact; a labelling convention, not a claim about the real timezone |
| 5 | **Prediction interval** | **1 hour** (`PT1H`), windows `[h, h+1h)`, **labelled by window start** | Base paper aggregates 1-min → 1-h; architecture "hourly forecast interval" |
| 6 | **Forecast horizon** | **1 step ahead** (the next hour). A forecast for hour `T` is issued once window `T−1h` has closed | Simplest horizon that supports rolling-error selection and drift on residuals |
| 7 | Seasonal period | 24 (hours) | SARIMA (1,0,1,24) in the base paper |
| 8 | Aggregation | mean of valid minutes for **every** column (so `sub_metering_*` become mean Wh/min) | Robust to missing minutes; keeps `target` in kW |
| 9 | Missing values | Raw minutes are **never imputed** (`is_missing=true`, value keys absent). An hour with no valid minute is carried forward (LOCF) and flagged `imputed=true` | Base paper uses last-observation fill; flagging keeps scoring honest |
| 10 | Event-time lateness | 300 s; a window closes when `max(event_time) − 300 s ≥ window_end`; later events are dropped and counted | Late/missing events handled by event time (abstract §6) |
| 11 | **Chronological split** | `train` < 2009-10-01 ≤ `val` < 2010-01-01 ≤ `test` (half-open, UTC) | train+val = base paper's 36-month training period; `test` = its 11-month validation period. `val` is for tuning/early stopping; refit on train+val before scoring on `test` is allowed |
| 12 | Replayed stream | `test` split by default (≈ 11 months, minute level) | Models never see the replayed period |
| 13 | Headline scoring | MAE/RMSE on `imputed == False` hours; also report all-hours | Long LOCF stretches would otherwise flatter or punish models arbitrarily |
| 14 | Identifiers | `run_id`, `meter_id` (Kafka key), `event_id`, `seq` — see §5 | Joins across topics and runs |

---

## 2. `energy.raw` — 1-minute reading (schema `energy_raw.v1`)

```json
{"schema_version":"1.0.0","run_id":"C_adaptive-r1","event_id":"uci-001-20100101T000000Z",
 "meter_id":"uci-001","event_time":"2010-01-01T00:00:00Z","seq":0,
 "replay_time_ms":1792522902047,"is_missing":false,
 "target":0.788,
 "features":{"global_reactive_power":0.224,"voltage":238.456,"global_intensity":3.3,
             "sub_metering_1":0.0,"sub_metering_2":0.0,"sub_metering_3":0.0}}
```

A missing minute (source `?`) — note that `target` and `features` are **absent**, not null:

```json
{"schema_version":"1.0.0","run_id":"C_adaptive-r1","event_id":"uci-001-20100105T124400Z",
 "meter_id":"uci-001","event_time":"2010-01-05T12:44:00Z","seq":9944,
 "replay_time_ms":1792522902990,"is_missing":true}
```

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | `"1.0.0"`. Consumers must reject unknown **major** versions |
| `run_id` | string `[A-Za-z0-9._-]{1,64}` | One id per experiment run, shared by every topic and every teammate's output |
| `event_id` | string | `<meter_id>-<event_time as YYYYMMDDTHHMMSSZ>`; deterministic, so redelivery overwrites instead of duplicating |
| `meter_id` | string | Kafka message key. `uci-001` is the real series; `uci-002…` are load replicas (identical values) |
| `event_time` | ISO-8601 UTC | **The** time axis for windowing, storage and joins |
| `seq` | int ≥ 0 | Per-run emission counter across all meters (`minute_index × n_meters + meter_index`); lets us count loss and duplication |
| `replay_time_ms` | int | Wall-clock epoch-ms when the producer emitted the record; basis of end-to-end latency |
| `is_missing` | bool | True if the source row had no target reading |
| `target` | number ≥ 0 | kW. Absent iff `is_missing` |
| `features` | object | Six keys above; a key is omitted if that single value was missing |

## 3. `energy.hourly` — forecast-interval aggregate (schema `energy_hourly.v1`) — *proposed new topic*

The architecture lists output topics but not the processed stream that the anomaly detector, drift monitor and forecasters all consume. I propose adding **`energy.hourly`** (produced by the event-time processor; reference implementation `cap-hourly`). Please confirm or rename.

```json
{"schema_version":"1.0.0","run_id":"C_adaptive-r1","event_id":"uci-001-H20100101T010000Z",
 "meter_id":"uci-001","event_time":"2010-01-01T01:00:00Z","window_end":"2010-01-01T02:00:00Z",
 "target":0.912,"features":{"global_reactive_power":0.19,"voltage":239.1,"global_intensity":3.9,
 "sub_metering_1":0.0,"sub_metering_2":0.4,"sub_metering_3":6.1},
 "n_valid":60,"n_expected":60,"imputed":false,
 "trigger_event_id":"uci-001-20100101T020500Z","trigger_replay_time_ms":1792522905110,
 "emit_time_ms":1792522905114}
```

`event_time` is the **window start**. `imputed` is true iff `n_valid == 0`. `trigger_*` names the raw event whose arrival closed the window (§4).

## 4. Rules for every downstream topic (`energy.predictions`, `.anomalies`, `.drift`, `.model_events`)

Payload fields are owned by Aleena (predictions, model_events) and Angel (anomalies, drift) — each publishes a JSON Schema into `src/capstone/schemas/` before integration. What is **frozen now** is the envelope, so the dashboard and evaluation can join everything:

| Envelope field | Rule |
|---|---|
| `schema_version`, `run_id`, `meter_id` | Copied from the input, never invented |
| `event_id` | Own deterministic id (suggest `<meter_id>-<kind>-<time>`) |
| `event_time` | The time **the data refers to**: start of the *predicted* hour for a forecast; the observation's own `event_time` for an anomaly; the window start at which drift was confirmed for a drift event. Never wall-clock |
| `trigger_event_id`, `trigger_replay_time_ms` | The upstream event that caused this output. Gives per-result end-to-end latency ("production to stored result", abstract §9) |
| `emit_time_ms` | Wall-clock epoch-ms at emission |

Also frozen: forecasts are joined to actuals on `(run_id, meter_id, event_time)`; anomaly flags describe **one observation**, drift describes **sustained change** and is never inferred from a single anomaly (architecture runtime rule); every component emits its latency/CPU/memory as `energy.metrics` events (`pipeline_metric.v1`) with its own `component` value.

## 5. Identifiers, topics, groups

| Item | Value |
|---|---|
| `meter_id` | `uci-001` (real), `uci-002…` replicas — **Kafka key** |
| `run_id` | `<config>-<repeat>`, e.g. `A_fixed_sarima-r1`, `B_fixed_lstm-r1`, `C_adaptive-r1`. Exactly one `runs/<run_id>/manifest.json` per run (§7) |
| Topics | `energy.raw`, `energy.hourly` *(proposed)*, `energy.predictions`, `energy.anomalies`, `energy.drift`, `energy.model_events`, `energy.metrics` |
| Partitions | `energy.raw`: set explicitly per run (experimental variable; default 3). Others: 1 |
| Consumer groups | `telegraf-base` (Telegraf → InfluxDB), `telegraf-metrics`, `adaptive-processor` (adaptive layer) — raw stream fans out to separate groups, as in the architecture |
| Delivery | At-least-once; every consumer must tolerate duplicates (deterministic `event_id`; InfluxDB overwrites identical series+timestamp) |

> **Why virtual meters?** UCI is one household. Keying by `meter_id` would put every record on one partition and make the partition/parallelism study meaningless. `cap-replay --meters N` replays the same series under N meter ids to scale volume and spread keys. Only `uci-001` is analysed for accuracy; replicas exist to load the pipeline.

## 6. Datasets (identical for all three of us — verify with `dataset_manifest.json`)

| File | Grain | Columns |
|---|---|---|
| `data/processed/uci_minute.parquet` | 1 min, **complete grid** | `event_time`, `target`, 6 features, `is_missing`, `split` |
| `data/processed/uci_hourly.parquet` | 1 h | `event_time` (window start), `window_end`, `target`, 6 features (hourly means), `n_valid`, `n_expected`, `imputed`, `split` |
| `data/processed/dataset_manifest.json` | — | row counts, **content SHA-256** per file, cut-offs |
| `data/processed/quality_report.md/json` | — | missingness, runs, imputed hours, ranges per split |

**Interface for anomaly/drift injection (Angel).** Produce an injected minute file with the **same columns and grid** as `uci_minute.parquet` (value columns modified, `is_missing=true` for injected drop-outs) plus a sidecar `labels.parquet`: `event_time`, `label` ∈ {`spike`,`drop`,`missing`,`level_shift`,`pattern_shift`}, `magnitude`, `segment_id`. Replay it with `cap-replay --source <file>`. **Ground-truth labels never travel on the stream.** Inject into the `test` split only.

## 7. Run manifest

`cap-replay` writes `runs/<run_id>/manifest.json` at start and completes it at the end: schema version, label, host, source slice, meters, full effective replay profile, Kafka settings incl. partition count, dataset content hashes, git commit, status. Two runs are comparable only if their manifests agree on everything except the variable under study. Do not hand-edit manifests.

## 8. InfluxDB mapping (base path)

`energy_raw` — tags `meter_id`, `run_id`; fields `target`, 6 features, `is_missing`, `seq`, `replay_time_ms`; timestamp = `event_time`. `pipeline_metric` — tags `run_id`, `component`, `name`, `instance`; field `value`; timestamp = `metric_time_ms`. Bucket retention is **infinite** because points carry 2006–2010 timestamps. Hot/cold: `cap-archive` moves points older than *N event-time days behind the stream watermark* to Hive-partitioned Parquet.

## 9. Change control and sign-off

Any change to §1–§8 = new `SCHEMA_VERSION` (patch: clarification; minor: additive optional field; major: anything else), a changelog line here, and a message to both other owners. Additive fields need no consumer changes; consumers must ignore nothing silently — `additionalProperties:false` is deliberate, so unknown fields fail validation in tests rather than in the demo.

- [x] Ananthan S — data / platform / integration
- [x] Aleena Antony — forecasting / selection
- [ ] Angel Treesa John — anomaly / drift

### Decisions I need a yes/no on before freezing

1. **Hourly aggregation = mean of valid minutes** (not sum/last). 
2. **LOCF without a maximum length**, flagged `imputed`. Alternative: cap at 24 h and exclude longer gaps from training/scoring.
3. **Three-way split** (train / val / test) rather than the paper's two-way — adds a tuning period without touching `test`.
4. **New `energy.hourly` topic** and the reference aggregator.
5. **Kafka default = Zookeeper** (baseline fidelity, as the abstract requires); KRaft is an optional compose profile for a separate modernisation run.
6. **UCI stamps labelled UTC, no DST handling.**
7. **Lateness = 300 s.**

### Changelog
- 1.0.0 — initial proposal.
