# Capstone Presentation Slides & Speaking Script
## Workstream 3: Streaming Anomaly Detection & Concept Drift Monitoring
**Presenter:** Angel Treesa John (Lead ML Systems Engineer)  
**Time Allocation:** ~3 to 4 minutes during team defense  

---

### Slide 1: Real-Time Anomaly Detection via Non-Parametric Rolling MAD

#### Slide Visuals & Bullet Points:
* **The Challenge**: Smart-meter telemetry exhibits extreme spikes (appliance turn-on) and sensor dropouts (inverter trips). Standard Z-scores ($\mu \pm 3\sigma$) break down because outliers inflate both the mean and variance.
* **Our Solution**:
  * **Non-Parametric Rolling Median & MAD**:
    $$\tilde{y} = \text{median}(y), \quad \text{MAD} = \text{median}(|y - \tilde{y}|)$$
    $$M = \frac{0.6745 \cdot |y - \tilde{y}|}{\text{MAD}}$$
  * **50% Breakdown Point**: Extreme surges never corrupt the baseline or mask subsequent faults.
  * **Automated Fault Classification**: Instantaneously categorizes `spike`, `drop`, `dropout` (inverter zero trip), and `missing` data.
* **Decoupling Rule (Core Architectural Guardrail)**:
  * Point anomalies trigger alerts to `energy.anomalies` for operators, but **never** trigger model retraining or router switching.
* **Benchmark Results**:
  * **Recall:** 100% | **Detection Delay:** 0 steps (instantaneous) | **Schema Compliance:** 100% Draft 2020-12.

#### Spoken Script for Angel (Slide 1):
> *"Hello everyone. In Workstream 3, I am responsible for Real-Time Anomaly Detection and Concept Drift Monitoring.*  
>  
> *In streaming energy systems, we face a major mathematical challenge: classical parametric detectors fail because extreme spikes inflate the sample mean and variance, causing masking effects. To solve this, I engineered a non-parametric rolling Median and Median Absolute Deviation detector. Because MAD has a 50% breakdown point, extreme surges cannot shift our baseline.*  
>  
> *Crucially, I enforced our team's core Decoupling Rule: single-point anomalies are published directly to Kafka topic `energy.anomalies` for monitoring, but they are strictly isolated from the forecasting models so they never trigger unwarranted retraining."*

---

### Slide 2: Adaptive Concept Drift Monitoring (ADWIN) & Dynamic Model Router Handoff

#### Slide Visuals & Bullet Points:
* **The Problem**: Energy consumption patterns structurally shift over time (seasonal transitions, new appliances, occupancy changes), leading to forecast degradation.
* **Adaptive Windowing (ADWIN)**:
  * Dynamically tracks streaming model residuals $e_t = |y_t - \hat{y}_t|$ in real time.
  * Continuously evaluates subwindow divergence against a rigorous statistical bound:
    $$\epsilon_{\text{cut}} = \sqrt{\frac{1}{2m} \cdot \ln\left( \frac{4n}{\delta} \right)}$$
* **Direct Integration with Dynamic Model Selector (Aleena's WS2)**:
  * When ADWIN confirms drift on Kafka topic `energy.drift`, it activates `Emergency Check 1: External Concept Drift Signal`.
  * Overrides routine dwell-time hysteresis to immediately trigger an emergency model re-evaluation and switch (e.g., LSTM $\rightarrow$ SARIMA/Lightweight fallback).
* **Live Ingestion & Visualization**:
  * Parsed via Telegraf `json_v2` into InfluxDB measurements `energy_anomalies` and `energy_drift`.
  * Visualized in Grafana (`anomaly_drift.json`) with live alert status indicators.
* **Drift Benchmark Results**:
  * **Shift Detection Rate:** 100% | **Mean Detection Delay:** 22 steps | **False Alarm Rate:** 0.013.

#### Spoken Script for Angel (Slide 2):
> *"While point anomalies represent isolated events, sustained forecast degradation indicates Concept Drift.*  
>  
> *To track this, I implemented the ADWIN algorithm over streaming prediction errors. When a statistically significant divergence between subwindows occurs, ADWIN flags a confirmed drift event on Kafka topic `energy.drift`.*  
>  
> *This interfaces directly with Aleena's Dynamic Model Selector: emitting a drift signal overrides normal dwell-time constraints, immediately triggering an emergency re-evaluation and model switch.*  
>  
> *Both output streams are parsed by Telegraf into InfluxDB and visualized on our Grafana dashboard. Our synthetic benchmark confirmed a 100% shift detection rate within 22 stream steps and an extremely low false alarm rate of 0.013. All 73 test suites across the capstone remain 100% green."*

---

### Slide 3: Quick Live Demonstration Checklist
1. **Show Grafana Dashboard (`energy-anomaly-drift`)**:
   * Highlight the power curve with real-time anomaly markers.
   * Point out the "Active Concept Drift Indicator" widget flipping from Green ("Stationary Regime") to Red ("Drift Alert Active").
2. **Show Test Suite Validation**:
   * Display `pytest` terminal output showing `73 passed in 13.75s`.
