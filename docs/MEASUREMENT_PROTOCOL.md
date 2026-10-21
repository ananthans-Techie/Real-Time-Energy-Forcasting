# Measurement protocol — pipeline metrics and the controlled A/B/C evaluation

Covers the "Measure throughput, end-to-end latency, consumer lag, CPU and memory" workstream item. All numbers land in `runs/<run_id>/` and `runs/summary_all.csv`; the same series are live in Grafana → *Pipeline – Health*.

## What is measured, and how

| Metric | Source | Definition / caveat |
|---|---|---|
| **Throughput** (records/s) | `producer_summary.json` (`achieved_rps`, `max_schedule_lag_s`); Telegraf `internal_write.metrics_written` rate | *Offered* load is the profile; *achieved* is delivered-acks / duration. `max_schedule_lag_s > ~1 s` means the producer itself fell behind its schedule — the run measures the producer, not the pipeline |
| **End-to-end latency** | `cap-probe` → `latency.csv`, `latency_summary.json` | Producer emit (`replay_time_ms`) → newest reading queryable in InfluxDB, sampled by polling (default 200 ms). Error ≤ poll interval + query time; **state the resolution next to every latency number**. Includes Kafka, Telegraf batching (`flush_interval`) and InfluxDB write |
| **Drain time** | `cap-probe` | Last record emitted → last record visible. Shows how much work is queued when the producer stops |
| **Consumer lag** | `cap-monitor` → `monitor.csv` (`kafka.consumer_lag`) | Σ over partitions of (log-end − committed offset) per group. Telegraf commits after InfluxDB accepts the batch, so lag includes the write |
| **CPU** | `cap-monitor` (`docker stats`) | Per container, 100 % = one core, can exceed 100 % |
| **Memory** | `cap-monitor` | Container working set (MiB) |
| Result latency for the adaptive layer | `trigger_replay_time_ms`, `emit_time_ms` on every downstream event | Per-result "production → stored result" (abstract §9) = stored-time − `trigger_replay_time_ms` |

## Controlled comparison (A fixed SARIMA, B fixed LSTM, C adaptive)

Hold constant: machine, Docker resource limits, dataset content hash, split (`test`), replay profile, `flush_interval`, Kafka partitions, measurement window. Vary only `run_id`/`label` and which model processes are running.

1. Fresh topics and a restarted Telegraf per run (`cap-benchmark` does both).
2. Discard the first 30 s of every time series as warm-up (JIT, consumer join, model load).
3. **≥ 3 repetitions** per configuration; report median and min–max, not a single run.
4. Record the `manifest.json` next to every table row; two rows are comparable only if their manifests differ in the studied variable alone.
5. Run the producer on the same host only if you say so: it competes for CPU. Prefer a separate machine for load > a few thousand records/s.

## Scenarios (`config/benchmark.yaml`)

| Scenario | Purpose | Expected reading |
|---|---|---|
| `baseline_500` | Base paper's rate (500 records/s) | Reference latency / CPU for A, B, C |
| `ramp_to_5000` | Start at 500 rec/s, +500 every 30 s | Rate at which lag first grows monotonically = saturation point of this deployment |
| `partitions_1/3/6` | Fixed offered load, 8 virtual meters | Effect of partitioning on lag and latency; a single Telegraf instance may limit gains — repeat with `docker compose up -d --scale telegraf=N` (N ≤ partitions) for the consumer-parallelism variant |
| fault / restart | `docker compose restart telegraf` (or kafka) mid-run | Lag spike and recovery time; at-least-once means duplicates are overwritten, check `seq` coverage in InfluxDB |

Forecast, anomaly, drift and adaptation metrics (MAE/RMSE, precision/recall/F1, detection delay, false alarms, recovery time, switching frequency) are owned by Aleena and Angel; they must compute them from outputs carrying the frozen envelope and the shared `run_id`.

## Known limitations (put these in the report)

* Single-broker, single-host Docker deployment: partition scaling shows software behaviour, not cluster behaviour.
* Virtual meters repeat identical values; they load the pipeline but add no modelling diversity.
* `docker stats` and the probe sample at ~1–2 s / 200 ms; sub-second spikes are invisible.
* Base paper's 0.39 s / 1.26 s figures were produced with a flush interval of 1 s on dedicated hardware (Xeon Silver 4210R, 32 GB) and a different workload definition; compare shapes (scaling behaviour), not absolute values.
