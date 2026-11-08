# Real-Time Anomaly Detection and Adaptive Concept Drift Monitoring in Streaming Energy Telemetry
## Capstone Research & Engineering Report — Workstream 3 Contribution

**Author:** Angel Treesa John (Lead ML Systems Engineer — Workstream 3)  
**Project:** Adaptive Real-Time Energy Pipeline Capstone  
**Supervisor / Reviewers:** Workstream 1 (Ananthan S), Workstream 2 (Aleena Antony)  
**Date:** October 2026  
**Document Version:** 1.0.0 — Final Implementation & Benchmark  

---

### Abstract
Modern distributed smart grids and industrial energy monitoring systems require high-throughput streaming intelligence capable of detecting sensor faults and demand surges in real time while adapting to non-stationary load patterns. In this project, Workstream 3 implements the real-time anomaly detection and concept drift monitoring layer of the Adaptive Real-Time Energy Pipeline. To avoid baseline corruption from extreme spikes, we introduce a non-parametric rolling Median and Median Absolute Deviation (MAD) detector operating under an event-driven Kafka architecture. Concurrently, an Adaptive Windowing (ADWIN) monitor tracks model forecast residuals to detect structural regime changes without confounding them with isolated point anomalies. When drift is confirmed, our engine directly triggers an emergency re-evaluation in the upstream Dynamic Model Selector. Benchmark evaluations demonstrate an anomaly detection recall of 100% with $\le 1$ step detection latency, and concept drift detection within 22 stream steps with a low false alarm rate of 0.013.

---

### 1. Introduction & Problem Formulation
In household and smart-grid energy telemetry, readings exhibit three distinct classes of irregularities:
1. **Point Anomalies (Spikes & Dips):** Transient surges caused by simultaneous appliance turn-on (e.g., water heater, AC compressor) or temporary power brownouts.
2. **Sensor & System Faults (Dropouts & Missing Gaps):** Zero readings from inverter trips or communications blackouts where telemetry is absent (`is_missing=True`).
3. **Concept Drift (Regime Changes):** Structural, sustained shifts in baseline energy consumption resulting from seasonal transitions, occupancy changes, or newly introduced electrical loads.

A critical design challenge in streaming analytics is **the decoupling rule**:
$$\text{Point Anomaly} \neq \text{Concept Drift}$$
A single extreme spike must trigger an anomaly alert on `energy.anomalies` for operational staff, but **must never** cause model retraining or dynamic model switching, which would destabilize forecasting accuracy. Conversely, sustained degradation in forecast residuals must be recognized as distribution drift on `energy.drift` to trigger dynamic model re-routing.

---

### 2. Methodology & Mathematical Formulations

#### 2.1 Non-Parametric Rolling Median and MAD Anomaly Detection
Classical parametric anomaly detectors rely on standard deviation $\sigma$ or Gaussian assumptions ($Z = \frac{x - \mu}{\sigma}$). In power load telemetry, extreme spikes severely inflate both the sample mean $\mu$ and standard deviation $\sigma$, causing masking effects where subsequent anomalies go undetected (0% breakdown point).

To solve this, we implement a rolling window of size $W = 24$ hours using the **Median Absolute Deviation (MAD)**, which has the maximum possible breakdown point of $50\%$:

1. **Rolling Median (Expected Value):**
   $$\tilde{y}_t = \text{median}(y_{t-W}, \dots, y_{t-1})$$

2. **Median Absolute Deviation (MAD):**
   $$\text{MAD}_t = \text{median}\left( \left| y_i - \tilde{y}_t \right| \right), \quad i \in [t-W, t-1]$$

3. **Modified Z-Score ($M_t$):**
   Under a normal distribution, $\text{E}[\text{MAD}] = \sigma / 1.4826 \approx 0.6745 \cdot \sigma$. Thus, the non-parametric modified Z-score is defined as:
   $$M_t = \frac{0.6745 \cdot \left| y_t - \tilde{y}_t \right|}{\text{MAD}_t}$$
   If $\text{MAD}_t \approx 0$ (e.g., during prolonged constant baseline), we fall back to the empirical standard deviation or a numerical floor $\epsilon = 10^{-5}$.

4. **Decision Boundary & Anomaly Classification:**
   An observation is flagged as an anomaly if $M_t \ge \tau$, where the default threshold $\tau = 3.5$ (Iglewicz & Hoaglin, 1993):
   $$\text{is\_anomaly}_t = \begin{cases} 
   \text{True}, & M_t \ge 3.5 \lor y_t = 0.0 \lor y_t \text{ is missing} \\
   \text{False}, & \text{otherwise}
   \end{cases}$$
   
   Classifications:
   * **`spike`**: $M_t \ge 3.5$ and $y_t > \tilde{y}_t$
   * **`drop`**: $M_t \ge 3.5$ and $y_t < \tilde{y}_t$
   * **`dropout`**: $y_t \le 0.001$ while $\tilde{y}_t > 0.4$ kW (inverter zero trip)
   * **`missing`**: Observation missing (`is_missing=True` or `target=None`)

5. **History Protection:** Points flagged as anomalous are excluded from the rolling baseline window, ensuring clean baseline estimation.

---

#### 2.2 Concept Drift Detection using Adaptive Windowing (ADWIN)
To detect distribution shifts and model degradation over the streaming prediction residuals, we implement ADWIN (Bifet & Gavaldà, 2007).

Let $e_t = |y_t - \hat{y}_t|$ be the absolute forecast error of the actively deployed model at hour $t$. ADWIN maintains a variable-size window $W$ of recent errors. For every partition of $W$ into two subwindows $W_0$ (older) and $W_1$ (newer) such that $W = W_0 \cdot W_1$:
* Let $n_0 = |W_0|$ and $n_1 = |W_1|$, with $n = n_0 + n_1$.
* Let $\mu_0$ and $\mu_1$ be the sample mean errors of $W_0$ and $W_1$.
* Harmonic mean of lengths:
  $$m = \frac{1}{1/n_0 + 1/n_1} = \frac{n_0 \cdot n_1}{n}$$
* Statistical cut bound at confidence parameter $\delta = 0.005$:
  $$\epsilon_{\text{cut}} = \sqrt{\frac{1}{2m} \cdot \ln\left( \frac{4n}{\delta} \right)}$$

**Detection Criterion:**
$$\text{If } |\mu_0 - \mu_1| \ge \epsilon_{\text{cut}} \implies \text{Concept Drift Confirmed}$$
Upon detection:
1. An event is emitted to `energy.drift` with `drift_detected=True`, `drift_score = |\mu_0 - \mu_1| / \epsilon_{\text{cut}}`, $\mu_0$, and $\mu_1$.
2. Older observations in $W_0$ are discarded, shrinking the window to adapt to the new regime.
3. The signal immediately forces Workstream 2's `DynamicModelSelector` to re-evaluate candidate models.

---

### 3. System Architecture & Kafka Integration

```
               [ energy.raw / energy.hourly ]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ AnomalyDetector ]               [ Streaming Forecaster ]
 (Rolling Median & MAD)               (LSTM / SARIMA Models)
            │                                 │
            ▼                                 ▼
   [ energy.anomalies ]             [ energy.predictions ]
   (Point alerts to DB)                       │
            │                         ┌───────┘
            │                         ▼
            │                 [ DriftMonitor ]
            │               (ADWIN on |y - y_hat|)
            │                         │
            │                         ▼
            │                  [ energy.drift ]
            │                         │
            │                         ▼
            └───────────────> [ Dynamic Selector ]
                             (Emergency Switch Trigger)
```

#### 3.1 Event Contracts & End-to-End Tracing
Every event emitted by Workstream 3 implements the shared envelope contract (`SCHEMA_CONTRACT_v1.0.0.md`):
* **Deterministic IDs**: `event_id = <meter_id>-A<time>` for anomalies, `<meter_id>-D<time>` for drift.
* **Lineage Preservation**: Inherits `trigger_event_id` and `trigger_replay_time_ms` from upstream hourly/prediction events.
* **Latency Tracing**: Downstream consumers compute end-to-end latency as:
  $$\text{Latency}_{\text{e2e}} = \text{emit\_time\_ms} - \text{trigger\_replay\_time\_ms}$$

#### 3.2 InfluxDB & Telegraf Ingestion
Both topics are ingested by Telegraf using `json_v2` parsers into dedicated InfluxDB measurements:
* `energy_anomalies`: Tags (`meter_id`, `run_id`, `anomaly_type`), Fields (`target_value`, `expected_value`, `anomaly_score`, `is_anomaly`, `threshold`).
* `energy_drift`: Tags (`meter_id`, `run_id`, `drift_method`), Fields (`drift_detected`, `drift_score`, `mean_error_before`, `mean_error_after`).

---

### 4. Synthetic Benchmark Suite & Experimental Results

#### 4.1 Synthetic Fault Injection Protocol
In compliance with Contract §6, fault injection is applied strictly to the `test` partition ($\ge \text{2010-01-01T00:00:00Z}$) with ground truth preserved in an isolated sidecar `labels.parquet`:
1. **Isolated Spikes**: Sudden +3.5 to +6.0 kW surges.
2. **Zero Dropouts**: Immediate drops to 0.0 kW.
3. **Missing Data Gaps**: Communication blackouts with `is_missing=True`.
4. **Sustained Level Shifts**: Step increase of +1.75 kW across 100+ hours.
5. **Variance Shifts**: $3.5\times$ amplification of oscillation amplitude.

#### 4.2 Benchmark Results Summary

| Component | Metric | Target | Achieved Result | Evaluation Notes |
|---|---|---|---|---|
| **Anomaly Detector** | Recall (Injected Faults) | $> 80.0\%$ | **100.0%** | Flagged all 7 labelled injected fault intervals in this benchmark |
| **Anomaly Detector** | Detection Latency | $\le 1$ step | **0.0 steps** | Instantaneous single-step classification |
| **Concept Drift** | Regime Shift Detection | $100\%$ | **100.0%** | Successfully identified sustained level shift and pattern shift |
| **Concept Drift** | Online Retrain Duration | $< 50$ ms | **4.20 ms** | Fast fine-tuning on historical regime windows |
| **Concept Drift** | False Alarm Rate | $< 0.05$ | **0.00189** | 15 alerts across 7,918 stream steps |
| **Platform** | Test Suite Status | $100\%$ Pass | **73 / 73 Green** | Full backwards compatibility with WS1 & WS2 |

#### 4.3 Controlled A/B/C Benchmark Comparison (Grand Capstone Evaluation)

| Architecture Configuration | Overall MAE (kW) | Normal Regime MAE | Post-Drift MAE (kW) | Error Degradation Under Drift |
| :--- | :---: | :---: | :---: | :---: |
| **Config A: Fixed SARIMA Baseline** | 0.2752 | 0.2747 | 0.5105 | +85.8% |
| **Config B: Fixed LSTM Baseline** | 0.2302 | 0.2292 | 0.7064 | +208.2% |
| **Config C: Full Adaptive Pipeline** | **0.2228** | **0.2224** | **0.3795** | **46.3% Error Reduction vs. LSTM** |

---

### 5. Conclusion
Workstream 3 delivers a robust, non-parametric, and decoupled streaming monitoring architecture. By combining Rolling MAD for point anomalies and ADWIN for forecast error distribution shifts, the pipeline guarantees real-time operational alerting without compromising predictive model stability.
