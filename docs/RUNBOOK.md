# Runbook

## First-time bring-up (≈10 min)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                       # change the secrets
make test                                   # unit + contract tests, no services needed
make prepare                                # downloads UCI, writes data/processed/*  (or: make synth-prepare to smoke-test first)
make up                                     # Zookeeper+Kafka, InfluxDB 2.7, Telegraf, Grafana
make topics RAW_PARTITIONS=3
make parser-check                           # prints the docker command; verifies the Telegraf mapping offline
make replay-smoke                           # 2000 minutes @ 500 rec/s
```

Then open Grafana <http://localhost:3000> (admin / admin by default) → *Energy Capstone* → **Energy – Readings**, set the time range to `2010-01-01 … 2010-02-01`, pick the run and meter.

## Verification checklist

| Check | Command | Pass |
|---|---|---|
| Topics exist | `cap-topics` | all 7 listed |
| Telegraf consuming | `docker compose logs --tail 20 telegraf` | no `E!` parse/connection errors |
| Points arriving | InfluxDB UI <http://localhost:8086> → Data Explorer → `energy_raw` | `target`, `is_missing`, `seq` present |
| No loss | stored `seq` count = `records_delivered` in `producer_summary.json` | equal |
| Missing minutes kept | `is_missing = true` points exist and have **no** `target` | yes |
| Lag drains | Grafana *Pipeline – Health* → consumer lag returns to 0 | yes |

## Hot → cold

```bash
cap-archive --run-id <run_id> --hot-days 30                 # copy to data/archive (Hive-partitioned Parquet)
cap-archive --run-id <run_id> --hot-days 30 --prune         # + verify counts, then delete from InfluxDB
```
"Old" is measured on **event time behind the run's newest point**, never wall-clock.
Reading cold data: `pd.read_parquet("data/archive/measurement=energy_raw/run_id=<id>")`. Grafana cannot read Parquet natively; either query hot data only, or add a DuckDB data-source plugin (optional, not wired here).

## Optional: KRaft instead of Zookeeper

`make up-kraft`. Run it as a *separate* experiment after the Zookeeper baseline, so that any difference is attributable (abstract §4 requires the modernisation not to obscure the baseline comparison).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Grafana: *no data* on Readings | Time range is wall-clock "last 6 h". Set it to the **event-time** period of the run |
| Grafana datasource error / Flux not supported | An InfluxDB **3** image was pulled. This repo pins `influxdb:2.7`; never use `:latest` |
| Telegraf logs `kafka: client has run out of available brokers` | Kafka not up yet (it retries), or `KAFKA_BOOTSTRAP` points at the host listener. Inside Docker use `kafka:19092`; on the host `localhost:9092` |
| Missing minutes absent from InfluxDB | Run `make parser-check`; if a field without `optional = true` fails on missing minutes, fix `telegraf.conf` |
| Old points vanish | Bucket retention is not infinite. Recreate the bucket with retention 0 |
| Lag grows right away at 500 rec/s | Producer + broker + Telegraf share one small VM; check CPU in *Pipeline – Health* before blaming a component |
| `cap-benchmark` hangs at probe | Telegraf not writing (see above); probe times out after `probe_timeout_s` |
| Re-running a scenario re-ingests old runs | New topic = new data; `cap-topics --recreate` clears the topic, and `cap-benchmark` does so per scenario |
