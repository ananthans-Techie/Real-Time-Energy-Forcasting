# Grand Capstone Defense Presentation
## Adaptive Electrical Energy Forecasting, Anomaly Detection & Concept Drift Monitoring
**Team Defense:** Ananthan S (WS1 Lead), Aleena Antony (WS2 Lead), Angel Treesa John (WS3 Lead)<br>
**Reference Architecture:** Im et al. (2024), *Frontiers in Big Data* (7:1308236)

---

### Slide 1: Executive Overview & The Problem of Non-Stationarity
* **The Problem**: In real-world electrical grids and residential meters, telemetry is highly dynamic:
  * Transient spikes from major appliances
  * Sensor dropouts and missing gaps
  * Structural concept drift (occupancy shifts, seasonal changes)
* **Failure of Static Models**: Traditional static baselines (SARIMA or fixed deep neural nets) degrade severely because they cannot adapt to non-stationary shifts.
* **Our Solution**:
  * Distributed event-driven streaming with Apache Kafka
  * Rolling 1-hour aggregation (`energy.hourly`) with watermark lateness handling
  * Multi-objective Dynamic Model Selection (SARIMA vs. LSTM vs. Lightweight)
  * Non-parametric Rolling MAD Anomaly Detection decoupled from ADWIN Concept Drift Monitoring
  * Autonomous online model fine-tuning and hot activation

---

### Slide 2: Streaming Architecture & Decoupled Intelligence
* **Unified Event Envelope**:
  * `trigger_event_id`, `trigger_replay_time_ms`, `emit_time_ms` tracked across every topic for strict latency tracing.
* **The Core Decoupling Rule**:
  $$\text{Point Anomaly (Spike/Dropout)} \neq \text{Concept Drift (Distribution Shift)}$$
  * Point anomalies (surges/inverter trips) are alerted on `energy.anomalies` but isolated from model training.
  * Concept drift is monitored on prediction residuals $e_t = |y_t - \hat{y}_t|$ via ADWIN.
* **Storage & Visualization Tier**:
  * Real-time metrics scraped via Telegraf into InfluxDB 2.7.
  * Live monitoring dashboards provisioned in Grafana 11.3.

---

### Slide 3: Controlled A/B/C Benchmark Experimental Findings
* **Dataset**: Evaluated on all 7,918 hours of the 2010 test split under identical chronological conditions.
* **Configuration Comparison**:
  * **Config A (Fixed SARIMA Baseline)**: Overall MAE = 0.2752 kW $\rightarrow$ Post-Drift MAE = 0.5105 kW (+85.8% error increase).
  * **Config B (Fixed LSTM Baseline)**: Overall MAE = 0.2302 kW $\rightarrow$ Post-Drift MAE = 0.7064 kW (+208.2% error surge).
  * **Config C (Full Adaptive Pipeline)**: Overall MAE = **0.2228 kW** $\rightarrow$ Post-Drift MAE = **0.3795 kW**.
* **Key Finding**:
  * The Adaptive Pipeline reduces post-drift forecast error by **46.3% vs. Fixed LSTM** and **25.7% vs. Fixed SARIMA**.

---

### Slide 4: Real-Time Detection Efficacy & Adaptation Latency
* **Anomaly Detection Performance**:
  * **Recall on Injected Faults**: 100.0% (Flagged all 7 labelled evaluation anomalies; applies specifically to this benchmark set).
  * **Precision & F1-Score**: 0.0012 and 0.0024 (reflects high sensitivity across 7,918 stream hours).
  * **Detection Latency**: 0.0 steps (instantaneous single-step classification at event arrival).
  * **Breakdown Point**: 50% (rolling MAD guarantees extreme spikes never shift baseline).
* **Concept Drift & Adaptation**:
  * ADWIN flagged 15 confirmed distribution shifts.
  * Triggered 15 online fine-tuning events on historical regime windows with zero future data leakage.
  * **Mean Retraining Duration**: **4.20 ms** (maximum 38.28 ms), ensuring zero pipeline lag.

---

### Slide 5: System Telemetry & Operational Production Readiness
* **Throughput**: **169.8 records/second** (well exceeding 60 rec/hr requirement).
* **Inference Latencies**:
  * **P50 Latency**: < 1.00 ms
  * **P95 Latency**: 1.79 ms
  * **P99 Latency**: 7.86 ms
* **Resource Footprint**:
  * Memory: 320.58 MiB RSS
  * CPU: 98.85% (single-core equivalent, vs. 608% for unconstrained PyTorch).
* **Quality Assurance**:
  * **73 / 73 automated unit and integration tests passing (100% green)**.
  * Schema Contract v1.0.0 validated across all 7 Kafka topics.
