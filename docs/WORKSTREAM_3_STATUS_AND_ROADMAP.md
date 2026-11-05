# Workstream 3: Anomaly Detection & Concept Drift
## Status, Deliverables, and Operational Roadmap
**Author:** Angel Treesa John (Lead ML Systems Engineer — Workstream 3)  
**Project:** Adaptive Real-Time Energy Pipeline Capstone  
**Version:** 1.0.0  
**Date:** October 2026  

---

### Executive Summary
Workstream 3 is responsible for **Real-Time Anomaly Detection**, **Concept Drift Monitoring**, **Output Kafka Topics (`energy.anomalies`, `energy.drift`)**, **Synthetic Evaluation Scenarios**, and **Integration Testing**.

This document outlines everything that has been engineered and validated in the codebase, alongside the remaining runtime and reporting tasks for the capstone demo and final report.

---

### Part 1: What Has Been Completed (100% Done & Verified)

#### 1. Schema Contracts & Platform Integration
- [x] **`energy.anomalies` Schema (`energy_anomalies.v1.schema.json`)**:
  - Validates `schema_version`, `run_id`, `event_id`, `meter_id`, `event_time`, `target_value`, `expected_value`, `anomaly_score`, `is_anomaly`, `anomaly_type`, `threshold`, and `window_size`.
  - Preserves end-to-end tracing fields (`trigger_event_id`, `trigger_replay_time_ms`, `emit_time_ms`).
- [x] **`energy.drift` Schema (`energy_drift.v1.schema.json`)**:
  - Validates `schema_version`, `run_id`, `event_id`, `meter_id`, `event_time`, `drift_detected`, `drift_score`, `mean_error_before`, `mean_error_after`, `drift_method`, and `window_size`.
- [x] **Team Contract Sign-off (`SCHEMA_CONTRACT_v1.0.0.md`)**:
  - Documented payload rules for §4.1 (`energy.anomalies`) and §4.2 (`energy.drift`).
  - Completed sign-off: `- [x] Angel Treesa John — anomaly / drift`.
- [x] **Shared Event Helper Functions (`src/capstone/events.py`)**:
  - Added schema mapping in `_KIND_TO_FILE` for `"anomaly"`, `"anomalies"`, and `"drift"`.
  - Added deterministic ID generators: `anomaly_event_id()` (`<meter_id>-A<time>`) and `drift_event_id()` (`<meter_id>-D<time>`).
  - Added typed builders: `build_anomaly_event()` and `build_drift_event()`.
- [x] **Telegraf & InfluxDB Configuration (`telegraf/telegraf.conf`)**:
  - Configured `inputs.kafka_consumer` with `data_format = "json_v2"` for both topics.
  - Maps to InfluxDB measurements: `energy_anomalies` and `energy_drift`.
  - Verified non-breaking compliance with `tests/test_platform_consistency.py`.

#### 2. Real-Time Anomaly Detection Module (`cap-anomaly` / `src/cap_anomaly/`)
- [x] **Non-Parametric Rolling Median & MAD (`RollingMADDetector`)**:
  - Calculates rolling baseline median $\tilde{y}$ and Median Absolute Deviation ($\text{MAD} = \text{median}(|y_i - \tilde{y}|)$).
  - Uses modified Z-score $M = \frac{0.6745 \cdot |y - \tilde{y}|}{\text{MAD}}$ with a 50% breakdown point to prevent extreme surges from shifting the baseline.
  - Classifies anomaly types: `spike` (power surge), `drop` (sharp dip), `dropout` (inverter zero trip), and `missing` (data gap).
- [x] **Point Anomaly Decoupling Rule**:
  - Strictly decouples transient point anomalies from sustained regime changes. Isolated spikes trigger immediate alerts on `energy.anomalies` without contaminating the baseline history or falsely triggering model retraining/switching.
- [x] **Streaming Kafka Consumer & Producer (`AnomalyDetector`)**:
  - Consumes from `energy.hourly` (or `energy.raw`).
  - Implements multi-meter state isolation (e.g. `uci-001`, `uci-002`).
  - Publishes validated UTF-8 JSON payloads to `energy.anomalies`.
  - CLI script registered: `cap-anomaly`.

#### 3. Concept Drift Monitoring Module (`cap-drift` / `src/cap_drift/`)
- [x] **ADWIN (Adaptive Windowing) Change Detector (`ADWIN`)**:
  - Standalone pure-Python/NumPy implementation of Bifet & Gavaldà (2007).
  - Dynamically cuts sliding windows upon detecting statistical divergence between subwindows: $\epsilon_{\text{cut}} = \sqrt{\frac{1}{2m} \ln\frac{4n}{\delta}}$.
- [x] **Page-Hinkley Sequential Test (`PageHinkley`)**:
  - Cumulative error tracking for rapid detection of abrupt residual shifts.
- [x] **Streaming Residual Monitor (`DriftMonitor`)**:
  - Subscribes to predictions (`energy.predictions`) and actual readings (`energy.hourly`).
  - Joins incoming streams on `(meter_id, event_time)` and calculates real-time absolute error $e_t = |y_t - \hat{y}_t|$.
  - Emits confirmed drift signals to `energy.drift`.
- [x] **Interface with Dynamic Model Selector (Workstream 2 Integration)**:
  - Confirmed drift signals feed directly into `DynamicModelSelector.evaluate_selection(..., drift_signal=True)`.
  - Bypasses normal hysteresis dwell time (`Emergency Check 1: External Concept Drift Signal`) to trigger an immediate model evaluation and router switch (e.g., LSTM $\rightarrow$ SARIMA/Lightweight).
  - CLI script registered: `cap-drift`.

#### 4. Synthetic Benchmark & Evaluation Suite (`scripts/` & `src/cap_anomaly/`)
- [x] **Synthetic Fault Injector (`SyntheticFaultInjector` / `scripts/inject_synthetic_events.py`)**:
  - Enforces Contract §6: Injects strictly into the `test` split ($\ge \text{2010-01-01T00:00:00Z}$) and preserves continuous time grid integrity.
  - Generates labeled scenarios:
    1. Single-point spikes (+3.5 to +6.0 kW)
    2. Zero-reading dropouts (0.0 kW)
    3. Communication blackout gaps (`is_missing=True`, missing values)
    4. Sustained step level shifts (+1.75 kW regime change)
    5. Multiplied variance / volatility ($3.5\times$ oscillation)
  - Produces isolated sidecar ground-truth `labels.parquet` (labels never travel over Kafka).
- [x] **Standardized Metrics Reporter (`AnomalyEvaluator` / `scripts/evaluate_workstream3.py`)**:
  - Anomaly metrics: Precision, Recall, F1-Score, Detection Delay (in stream steps).
  - Drift metrics: Mean Detection Delay (steps to flag sustained shift), False Alarm Rate (false positives per stationary step).
  - Automated Markdown evaluation report generator (`runs/workstream3_evaluation_report.md`).

#### 5. Automated Test Suite (100% Green)
- [x] **`tests/test_anomaly.py`** (7 tests): Rolling MAD, spike isolation, dropout, missing values, schema validation, multi-meter isolation.
- [x] **`tests/test_drift.py`** (6 tests): ADWIN stationary stability, sustained shift detection, Page-Hinkley, stream matching, DynamicModelSelector handoff.
- [x] **`tests/test_synthetic_injector.py`** (4 tests): Injected dataset grid, sidecar schema, metric computation, Markdown reporting.
- [x] **Full Suite Status**: **73 / 73 passing** (56 existing from WS1 & WS2 + 17 new WS3 tests).

---

### Part 2: What is Remaining to Do in Your Part (Roadmap & Next Steps)

While the entire core codebase, streaming modules, schemas, and test suites are complete, the following runtime, infrastructure, and presentation tasks remain for the live system demonstration and final project report:

#### Phase 1: End-to-End Live Pipeline Demo (with Docker & Kafka)
1. **Live Containerized Ingestion Check**:
   - Spin up Docker services: `docker compose up -d zookeeper kafka telegraf influxdb grafana`.
   - Create Kafka topics: `cap-topics --raw-partitions 3`.
   - Replay real or synthetic injected data:
     ```bash
     cap-replay --source data/processed/injected_minute.parquet --profile max
     ```
   - Verify that `cap-anomaly` consumes `energy.hourly` and produces to `energy.anomalies`.
   - Verify that `cap-drift` consumes `energy.predictions` + `energy.hourly` and produces to `energy.drift`.

2. **InfluxDB Data Ingestion Verification**:
   - Query InfluxDB v2 bucket to confirm that Telegraf is writing points into:
     - Measurement `energy_anomalies`
     - Measurement `energy_drift`

#### Phase 2: Grafana Dashboard Panel Integration
1. **Add Anomaly & Drift Panels to Grafana**:
   - Workstream 1 created `grafana/dashboards/energy_readings.json`.
   - Add visualization panels:
     - **Time Series / Points Panel**: Overlay detected point anomalies (spikes/dropouts) on the raw energy power curve.
     - **Status / Alert Indicator**: Visual indicator for active concept drift status (Normal vs. Drift Detected).
     - **Residual Error with ADWIN Cut Line**: Plot showing $|y - \hat{y}|$ and the adaptive threshold.

#### Phase 3: Final Controlled Benchmark Experiments
1. **Run Evaluation on Full UCI Dataset**:
   - Download/prepare full UCI dataset using `cap-prepare` (if not already cached).
   - Inject benchmark suite into the full 11-month test period using `python scripts/inject_synthetic_events.py`.
   - Run `python scripts/evaluate_workstream3.py` on the full test split to record headline metrics:
     - Anomaly Detection: Target Precision $> 0.85$, Recall $> 0.80$, F1 $> 0.82$, Delay $\le 1$ step.
     - Concept Drift: Target Shift Detection $100\%$, Delay $< 35$ steps, False Alarm Rate $< 0.05$.

#### Phase 4: Capstone Final Report & Presentation Contributions
1. **Section Write-ups for Final Report**:
   - **Methodology §3.3**: Mathematical formulation of Rolling MAD (modified Z-score) and ADWIN harmonic subwindow variance testing.
   - **Architecture §4.3**: End-to-end tracing lineage (`trigger_event_id` $\rightarrow$ latency), Kafka topic topology, and handoff to Workstream 2's Dynamic Selector.
   - **Experimental Results §5.3**: Tables and charts showing precision, recall, detection latency, and recovery time after drift.
2. **Slide Deck & Demonstration Video**:
   - Prepare a 2-slide summary for Workstream 3 highlighting:
     - Why non-parametric MAD was chosen over static Z-score / IQR.
     - How ADWIN detects regime change and triggers emergency model routing.
     - Video capture of Grafana dashboard updating in real time during a synthetic fault injection.

---

### Part 3: Quick Reference of Workstream 3 Artifacts

| Component | File Path | Status |
|---|---|---|
| **Anomaly Schema** | `src/capstone/schemas/energy_anomalies.v1.schema.json` | Complete |
| **Drift Schema** | `src/capstone/schemas/energy_drift.v1.schema.json` | Complete |
| **Contract** | `SCHEMA_CONTRACT_v1.0.0.md` | Signed off |
| **Anomaly Module** | `src/cap_anomaly/anomaly_detector.py` | Complete (`cap-anomaly`) |
| **ADWIN / Page-Hinkley** | `src/cap_drift/adwin.py` | Complete |
| **Drift Monitor** | `src/cap_drift/drift_monitor.py` | Complete (`cap-drift`) |
| **Fault Injector** | `src/cap_anomaly/synthetic_injector.py` | Complete (`cap-inject`) |
| **Metrics Evaluator** | `src/cap_anomaly/evaluator.py` | Complete |
| **CLI Runner: Inject** | `scripts/inject_synthetic_events.py` | Complete |
| **CLI Runner: Evaluate** | `scripts/evaluate_workstream3.py` | Complete |
| **Telegraf Config** | `telegraf/telegraf.conf` | Complete |
| **Unit & Integration Tests** | `tests/test_anomaly.py`, `test_drift.py`, `test_synthetic_injector.py` | 17/17 Passed |
