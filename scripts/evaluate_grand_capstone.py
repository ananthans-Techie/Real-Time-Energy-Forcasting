"""Grand Capstone Comprehensive Evaluation & Benchmarking Suite.

Executes:
  - Configuration A: Fixed SARIMA Baseline (no adaptation)
  - Configuration B: Fixed LSTM Baseline (no adaptation)
  - Configuration C: Full Adaptive Pipeline (Dynamic Selection + ADWIN Drift Detection + Retraining Adaptation)

Computes:
  - MAE, RMSE, MAPE (segmented: Overall, Normal hours, Post-drift hours)
  - Precision, Recall, F1, Detection Delay (Anomalies & Concept Drift)
  - System Telemetry: Throughput, Latencies (P50, P95, P99), Resource Footprint (CPU %, Memory MiB)

Generates 10 Publication-Quality Figures into runs/figures/ and writes runs/benchmark_abc_metrics.json.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

# Ensure matplotlib cache is in a local directory
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / ".matplotlib_cache")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns

from cap_anomaly import AnomalyDetector, AnomalyEvaluator
from cap_drift import DriftMonitor
from capstone import constants as C
from capstone.events import build_model_event
from capstone.forecasting.forecaster import StreamingForecaster
from capstone.forecasting.models import LSTMForecaster, SARIMAForecaster, LightweightForecaster
from capstone.forecasting.selector import DynamicModelSelector, ModelPerformance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("grand_evaluation")

# Publication plot styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def calculate_forecasting_metrics(actuals: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and zero-safe MAPE."""
    diff = preds - actuals
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Safe MAPE: avoid division by zero using small epsilon
    eps = 1e-3
    denom = np.maximum(np.abs(actuals), eps)
    mape = float(np.mean(np.abs(diff) / denom) * 100.0)
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
    }


def run_evaluation(
    hourly_path: str = "data/processed/injected_hourly.parquet",
    labels_path: str = "data/processed/labels.parquet",
    out_dir: str = "runs",
    eval_limit_hours: int | None = None,
) -> dict[str, Any]:
    out_p = Path(out_dir)
    figures_p = out_p / "figures"
    figures_p.mkdir(parents=True, exist_ok=True)

    logger.info("Loading test dataset and ground truth labels...")
    df_all = pd.read_parquet(hourly_path)
    test_df = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)
    if eval_limit_hours and len(test_df) > eval_limit_hours:
        test_df = test_df.iloc[:eval_limit_hours].copy()

    labels_df = pd.read_parquet(labels_path)
    labels_df["event_time"] = pd.to_datetime(labels_df["event_time"], utc=True)
    test_df["event_time"] = pd.to_datetime(test_df["event_time"], utc=True)

    # Hourly ground-truth labels for anomaly and drift evaluation
    labels_hourly = labels_df.copy()
    labels_hourly["event_time"] = labels_hourly["event_time"].dt.floor("h")
    labels_hourly = labels_hourly.drop_duplicates(subset=["event_time", "label"]).reset_index(drop=True)

    # Identify time periods: Normal vs Drift / Post-Drift
    drift_times = set(labels_hourly[labels_hourly["label"].isin(["level_shift", "pattern_shift"])]["event_time"])
    is_drift_mask = test_df["event_time"].isin(drift_times).to_numpy()

    logger.info(f"Loaded {len(test_df):,} test hours. Drift hours: {is_drift_mask.sum():,}, Normal hours: {(~is_drift_mask).sum():,}")

    actuals = test_df[C.TARGET].to_numpy(dtype=float)
    event_times = test_df["event_time"].tolist()
    iso_times = [t.strftime("%Y-%m-%dT%H:00:00Z") for t in event_times]

    proc = psutil.Process()

    # =========================================================================
    # Configuration A: Fixed SARIMA Baseline (no adaptation)
    # =========================================================================
    logger.info("=== Running Configuration A: Fixed SARIMA ===")
    forecaster_a = StreamingForecaster.from_directory(models_dir="models", fixed_model="sarima")
    preds_a = []
    lats_a = []
    cpu_samples_a = []
    mem_samples_a = []
    proc.cpu_percent(None)

    t0 = time.time()
    for i, row in test_df.iterrows():
        t_iso = iso_times[i]
        next_t_iso = (row["event_time"] + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00:00Z")
        raw_eid = t_iso.replace("-", "").replace(":", "")
        ev = {
            "schema_version": "1.0.0",
            "run_id": "eval-config-A",
            "event_id": f"uci-001-H{raw_eid}",
            "meter_id": "uci-001",
            "event_time": t_iso,
            "window_end": next_t_iso,
            "target": float(row[C.TARGET]),
            "features": {f: float(row[f]) if f in row and not pd.isna(row[f]) else 0.0 for f in C.FEATURES},
            "n_valid": 60,
            "n_expected": 60,
            "imputed": bool(row.get("imputed", False)),
            "trigger_event_id": f"trig-A-{i}",
            "trigger_replay_time_ms": 1000,
            "emit_time_ms": int(time.time() * 1000),
        }
        pred_ev, _ = forecaster_a.process_hourly_event(ev, drift_signal=False)
        preds_a.append(pred_ev["prediction"])
        lats_a.append(pred_ev["inference_latency_ms"])

        if i % 100 == 0:
            cpu_samples_a.append(proc.cpu_percent(None))
            mem_samples_a.append(proc.memory_info().rss / (1024 * 1024))

    duration_a = time.time() - t0
    preds_a = np.array(preds_a)

    # =========================================================================
    # Configuration B: Fixed LSTM Baseline (no adaptation)
    # =========================================================================
    logger.info("=== Running Configuration B: Fixed LSTM ===")
    forecaster_b = StreamingForecaster.from_directory(models_dir="models", fixed_model="lstm")
    preds_b = []
    lats_b = []
    cpu_samples_b = []
    mem_samples_b = []
    proc.cpu_percent(None)

    t0 = time.time()
    for i, row in test_df.iterrows():
        t_iso = iso_times[i]
        next_t_iso = (row["event_time"] + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00:00Z")
        raw_eid = t_iso.replace("-", "").replace(":", "")
        ev = {
            "schema_version": "1.0.0",
            "run_id": "eval-config-B",
            "event_id": f"uci-001-H{raw_eid}",
            "meter_id": "uci-001",
            "event_time": t_iso,
            "window_end": next_t_iso,
            "target": float(row[C.TARGET]),
            "features": {f: float(row[f]) if f in row and not pd.isna(row[f]) else 0.0 for f in C.FEATURES},
            "n_valid": 60,
            "n_expected": 60,
            "imputed": bool(row.get("imputed", False)),
            "trigger_event_id": f"trig-B-{i}",
            "trigger_replay_time_ms": 1000,
            "emit_time_ms": int(time.time() * 1000),
        }
        pred_ev, _ = forecaster_b.process_hourly_event(ev, drift_signal=False)
        preds_b.append(pred_ev["prediction"])
        lats_b.append(pred_ev["inference_latency_ms"])

        if i % 100 == 0:
            cpu_samples_b.append(proc.cpu_percent(None))
            mem_samples_b.append(proc.memory_info().rss / (1024 * 1024))

    duration_b = time.time() - t0
    preds_b = np.array(preds_b)

    # =========================================================================
    # Configuration C: Full Adaptive Pipeline (Dynamic Selection + Drift + Retrain)
    # =========================================================================
    logger.info("=== Running Configuration C: Full Adaptive Pipeline ===")
    forecaster_c = StreamingForecaster.from_directory(
        models_dir="models",
        default_model="lstm",
        candidate_models=["lstm", "sarima", "lightweight"],
    )
    anomaly_detector = AnomalyDetector(window_size=24, threshold=3.5)
    drift_monitor = DriftMonitor(method="adwin", delta=0.005)

    preds_c = []
    lats_c = []
    active_models_c = []
    model_events_c = []
    anomalies_c = []
    drifts_c = []
    drift_scores_c = []
    retrain_events = []
    cpu_samples_c = []
    mem_samples_c = []
    proc.cpu_percent(None)

    pending_drift_signal = False

    t0 = time.time()
    for i, row in test_df.iterrows():
        t_iso = iso_times[i]
        next_t_iso = (row["event_time"] + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00:00Z")
        target_v = float(row[C.TARGET])
        raw_eid = t_iso.replace("-", "").replace(":", "")

        # 1. Anomaly detection on incoming reading
        rec_anom = {
            "run_id": "eval-config-C",
            "event_id": f"uci-001-H{raw_eid}",
            "meter_id": "uci-001",
            "event_time": t_iso,
            "target": target_v,
            "is_missing": bool(row.get("is_missing", False)),
            "trigger_replay_time_ms": 1000,
        }
        anom_ev = anomaly_detector.process_record(rec_anom)
        if anom_ev:
            anomalies_c.append(anom_ev)

        # 2. Concept drift monitoring against previous prediction
        if len(preds_c) > 0:
            residual = abs(target_v - preds_c[-1])
            drift_ev = drift_monitor.process_error(
                run_id="eval-config-C",
                meter_id="uci-001",
                event_time_iso=t_iso,
                error_val=residual,
                trigger_event_id=rec_anom["event_id"],
                trigger_replay_time_ms=1000,
            )
            drifts_c.append(drift_ev)
            drift_scores_c.append(drift_ev["drift_score"])
            if drift_ev["drift_detected"]:
                pending_drift_signal = True
                logger.info(f"ADWIN Drift confirmed at step {i} ({t_iso}), score={drift_ev['drift_score']:.3f}")
        else:
            drift_scores_c.append(0.0)

        # 3. Handle Drift-Triggered Online Model Adaptation & Retraining
        if pending_drift_signal:
            prev_m = forecaster_c.selector.active_model
            retrain_start = time.time()
            # Retrain adaptive model strictly on historical regime window (zero future leakage)
            regime_window = test_df.iloc[max(0, i - 72) : i + 1]
            adapted_model = LightweightForecaster(lags=(1, 2, 24))
            adapted_model.fit(regime_window)
            retrain_dur_ms = (time.time() - retrain_start) * 1000.0

            adapted_name = "adapted_regime"
            forecaster_c.models[adapted_name] = adapted_model
            if adapted_name not in forecaster_c.selector.candidate_models:
                forecaster_c.selector.candidate_models.append(adapted_name)
                forecaster_c.selector.trackers[adapted_name] = ModelPerformance()
                forecaster_c.selector.resource_costs[adapted_name] = 0.10

            forecaster_c.selector.active_model = adapted_name
            forecaster_c.selector.hours_since_switch = 0

            retrain_ev = build_model_event(
                run_id="eval-config-C",
                meter_id="uci-001",
                event_time_iso=next_t_iso,
                event_type="retrain",
                previous_model=prev_m,
                new_model=adapted_name,
                trigger_metric="drift_signal",
                trigger_value=1.0,
                reason=f"ADWIN concept drift confirmed. Retrained and activated {adapted_name} in {retrain_dur_ms:.2f}ms.",
                metrics={"retrain_duration_ms": retrain_dur_ms, "step": i},
            )
            model_events_c.append(retrain_ev)
            retrain_events.append(retrain_ev)
            logger.info(f"Adaptation complete: Activated '{adapted_name}' (duration: {retrain_dur_ms:.2f}ms)")
            pending_drift_signal = False

        # 4. Process hourly forecasting event
        ev = {
            "schema_version": "1.0.0",
            "run_id": "eval-config-C",
            "event_id": f"uci-001-H{raw_eid}",
            "meter_id": "uci-001",
            "event_time": t_iso,
            "window_end": next_t_iso,
            "target": target_v,
            "features": {f: float(row[f]) if f in row and not pd.isna(row[f]) else 0.0 for f in C.FEATURES},
            "n_valid": 60,
            "n_expected": 60,
            "imputed": bool(row.get("imputed", False)),
            "trigger_event_id": f"trig-C-{i}",
            "trigger_replay_time_ms": 1000,
            "emit_time_ms": int(time.time() * 1000),
        }
        pred_ev, model_ev = forecaster_c.process_hourly_event(ev, drift_signal=False)
        preds_c.append(pred_ev["prediction"])
        lats_c.append(pred_ev["inference_latency_ms"])
        active_models_c.append(forecaster_c.selector.active_model)
        if model_ev:
            model_events_c.append(model_ev)

        if i % 100 == 0:
            cpu_samples_c.append(proc.cpu_percent(None))
            mem_samples_c.append(proc.memory_info().rss / (1024 * 1024))

    duration_c = time.time() - t0
    preds_c = np.array(preds_c)

    # =========================================================================
    # Phase 5: Metrics Calculation
    # =========================================================================
    logger.info("=== Calculating Comparative Metrics ===")

    # 1. ML Accuracy (Overall, Normal, Post-Drift)
    metrics_a = {
        "overall": calculate_forecasting_metrics(actuals, preds_a),
        "normal": calculate_forecasting_metrics(actuals[~is_drift_mask], preds_a[~is_drift_mask]),
        "post_drift": calculate_forecasting_metrics(actuals[is_drift_mask], preds_a[is_drift_mask]),
    }
    metrics_b = {
        "overall": calculate_forecasting_metrics(actuals, preds_b),
        "normal": calculate_forecasting_metrics(actuals[~is_drift_mask], preds_b[~is_drift_mask]),
        "post_drift": calculate_forecasting_metrics(actuals[is_drift_mask], preds_b[is_drift_mask]),
    }
    metrics_c = {
        "overall": calculate_forecasting_metrics(actuals, preds_c),
        "normal": calculate_forecasting_metrics(actuals[~is_drift_mask], preds_c[~is_drift_mask]),
        "post_drift": calculate_forecasting_metrics(actuals[is_drift_mask], preds_c[is_drift_mask]),
    }

    # 2. Detection Efficacy Metrics
    anomaly_eval = AnomalyEvaluator.evaluate_anomalies(labels_hourly, anomalies_c)
    drift_eval = AnomalyEvaluator.evaluate_drift(labels_hourly, drifts_c, stream_event_times=test_df["event_time"])

    # 3. System Computational Performance
    sys_perf = {
        "config_A": {
            "throughput_rps": round(len(test_df) / duration_a, 1),
            "p50_latency_ms": round(float(np.percentile(lats_a, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(lats_a, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(lats_a, 99)), 2),
            "cpu_mean_pct": round(float(np.mean(cpu_samples_a)) if cpu_samples_a else 0.0, 2),
            "mem_mib": round(float(np.mean(mem_samples_a)) if mem_samples_a else 0.0, 2),
            "execution_duration_sec": round(duration_a, 2),
        },
        "config_B": {
            "throughput_rps": round(len(test_df) / duration_b, 1),
            "p50_latency_ms": round(float(np.percentile(lats_b, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(lats_b, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(lats_b, 99)), 2),
            "cpu_mean_pct": round(float(np.mean(cpu_samples_b)) if cpu_samples_b else 0.0, 2),
            "mem_mib": round(float(np.mean(mem_samples_b)) if mem_samples_b else 0.0, 2),
            "execution_duration_sec": round(duration_b, 2),
        },
        "config_C": {
            "throughput_rps": round(len(test_df) / duration_c, 1),
            "p50_latency_ms": round(float(np.percentile(lats_c, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(lats_c, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(lats_c, 99)), 2),
            "cpu_mean_pct": round(float(np.mean(cpu_samples_c)) if cpu_samples_c else 0.0, 2),
            "mem_mib": round(float(np.mean(mem_samples_c)) if mem_samples_c else 0.0, 2),
            "execution_duration_sec": round(duration_c, 2),
        },
    }

    results = {
        "dataset_rows_evaluated": len(test_df),
        "evaluation_period": f"{iso_times[0]} to {iso_times[-1]}",
        "accuracy": {
            "config_A_fixed_sarima": metrics_a,
            "config_B_fixed_lstm": metrics_b,
            "config_C_full_adaptive": metrics_c,
        },
        "anomaly_detection": anomaly_eval,
        "concept_drift_detection": drift_eval,
        "system_performance": sys_perf,
        "model_adaptations": [
            {
                "event_time": ev["event_time"],
                "event_type": ev["event_type"],
                "previous_model": ev.get("previous_model"),
                "new_model": ev["new_model"],
                "trigger_metric": ev["trigger_metric"],
                "reason": ev["reason"],
            }
            for ev in retrain_events
        ],
    }

    # Save JSON summary
    metrics_file = out_p / "benchmark_abc_metrics.json"
    metrics_file.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved machine-readable benchmark metrics -> {metrics_file}")

    # Save raw predictions and event logs for full reproducibility
    pd.DataFrame({"event_time": event_times, "actual": actuals, "prediction": preds_a, "latency_ms": lats_a}).to_parquet(out_p / "config_A_predictions.parquet")
    pd.DataFrame({"event_time": event_times, "actual": actuals, "prediction": preds_b, "latency_ms": lats_b}).to_parquet(out_p / "config_B_predictions.parquet")
    pd.DataFrame({
        "event_time": event_times,
        "actual": actuals,
        "prediction": preds_c,
        "active_model": active_models_c,
        "latency_ms": lats_c,
        "drift_score": drift_scores_c,
    }).to_parquet(out_p / "config_C_predictions.parquet")

    (out_p / "config_C_model_events.json").write_text(json.dumps(model_events_c, indent=2))
    (out_p / "config_C_drift_events.json").write_text(json.dumps(drifts_c, indent=2))
    (out_p / "config_C_anomaly_events.json").write_text(json.dumps(anomalies_c, indent=2))

    # Summary tabular CSV
    summary_rows = [
        {"configuration": "Config A (Fixed SARIMA)", "regime": "Overall", **metrics_a["overall"], **sys_perf["config_A"]},
        {"configuration": "Config A (Fixed SARIMA)", "regime": "Normal", **metrics_a["normal"]},
        {"configuration": "Config A (Fixed SARIMA)", "regime": "Post-Drift", **metrics_a["post_drift"]},
        {"configuration": "Config B (Fixed LSTM)", "regime": "Overall", **metrics_b["overall"], **sys_perf["config_B"]},
        {"configuration": "Config B (Fixed LSTM)", "regime": "Normal", **metrics_b["normal"]},
        {"configuration": "Config B (Fixed LSTM)", "regime": "Post-Drift", **metrics_b["post_drift"]},
        {"configuration": "Config C (Full Adaptive)", "regime": "Overall", **metrics_c["overall"], **sys_perf["config_C"]},
        {"configuration": "Config C (Full Adaptive)", "regime": "Normal", **metrics_c["normal"]},
        {"configuration": "Config C (Full Adaptive)", "regime": "Post-Drift", **metrics_c["post_drift"]},
    ]
    pd.DataFrame(summary_rows).to_csv(out_p / "benchmark_summary.csv", index=False)
    logger.info(f"Saved benchmark summary table -> {out_p / 'benchmark_summary.csv'}")

    # =========================================================================
    # Phase 7: Generate Publication-Quality Figures (1 to 10)
    # =========================================================================
    logger.info("=== Rendering Publication Visualizations (10 Figures) ===")
    steps = np.arange(len(test_df))

    # Figure 1: Actual vs Predicted Energy Consumption (Initial 200 hours)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps[:200], actuals[:200], label="Actual Telemetry (kW)", color="#1f2937", lw=1.8)
    ax.plot(steps[:200], preds_c[:200], label="Adaptive Forecast (kW)", color="#2563eb", ls="--", lw=1.5)
    ax.set_title("Figure 1: Actual vs. Predicted Energy Consumption (1-Step Ahead)", fontweight="bold")
    ax.set_xlabel("Hourly Prediction Window (Steps)")
    ax.set_ylabel("Power Demand (kW)")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_p / "fig01_actual_vs_predicted.png")
    plt.close()

    # Figure 2: SARIMA vs LSTM vs Adaptive Forecasts (Zoomed during Level Shift Regime)
    fig, ax = plt.subplots(figsize=(13, 5))
    zoom = slice(3940, 4000) if len(steps) >= 4000 else slice(0, min(150, len(steps)))
    ax.plot(steps[zoom], actuals[zoom], label="Actual Load (kW)", color="#111827", lw=2.0)
    ax.plot(steps[zoom], preds_a[zoom], label="Fixed SARIMA (Config A)", color="#dc2626", ls=":", lw=1.5)
    ax.plot(steps[zoom], preds_b[zoom], label="Fixed LSTM (Config B)", color="#d97706", ls="-.", lw=1.5)
    ax.plot(steps[zoom], preds_c[zoom], label="Full Adaptive (Config C)", color="#16a34a", ls="-", lw=2.0)
    ax.set_title("Figure 2: Forecast Comparison During Concept Drift Regime Shift", fontweight="bold")
    ax.set_xlabel("Hourly Prediction Window (Steps)")
    ax.set_ylabel("Power Demand (kW)")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_p / "fig02_sarima_vs_lstm_vs_adaptive.png")
    plt.close()

    # Figure 3: Normal vs Post-Drift Forecasting Errors
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["Config A\n(Fixed SARIMA)", "Config B\n(Fixed LSTM)", "Config C\n(Full Adaptive)"]
    x = np.arange(len(labels))
    width = 0.35
    normal_maes = [metrics_a["normal"]["mae"], metrics_b["normal"]["mae"], metrics_c["normal"]["mae"]]
    drift_maes = [metrics_a["post_drift"]["mae"], metrics_b["post_drift"]["mae"], metrics_c["post_drift"]["mae"]]
    ax.bar(x - width/2, normal_maes, width, label="Normal Regime MAE", color="#3b82f6")
    ax.bar(x + width/2, drift_maes, width, label="Post-Drift Regime MAE", color="#ef4444")
    ax.set_title("Figure 3: Forecasting Error Degradation (Normal vs. Post-Drift)", fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (kW)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True)
    for i in range(len(labels)):
        ax.annotate(f"{normal_maes[i]:.2f}", (x[i] - width/2, normal_maes[i] + 0.02), ha="center", fontsize=9)
        ax.annotate(f"{drift_maes[i]:.2f}", (x[i] + width/2, drift_maes[i] + 0.02), ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_p / "fig03_normal_vs_postdrift_errors.png")
    plt.close()

    # Figure 4: Anomaly Detection Timeline (Spike at index 395)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    anom_slice = slice(350, 450) if len(steps) >= 450 else slice(0, len(steps))
    steps_anom = steps[anom_slice]
    act_anom = actuals[anom_slice]
    ax1.plot(steps_anom, act_anom, color="#4b5563", lw=1.2, label="Actual Power (kW)")
    flagged_steps = [s for s in steps_anom if s < len(anomalies_c) and anomalies_c[s]["is_anomaly"]]
    flagged_vals = [actuals[s] for s in flagged_steps]
    ax1.scatter(flagged_steps, flagged_vals, color="#dc2626", s=45, zorder=5, label="Flagged Anomaly")
    ax1.set_ylabel("Power (kW)")
    ax1.set_title("Figure 4: Real-Time Anomaly Detection Timeline (Rolling MAD)", fontweight="bold")
    ax1.legend(loc="upper right")

    scores_slice = [anomalies_c[s]["anomaly_score"] if s < len(anomalies_c) else 0.0 for s in steps_anom]
    ax2.plot(steps_anom, scores_slice, color="#8b5cf6", lw=1.2, label="Modified Z-Score")
    ax2.axhline(3.5, color="#dc2626", ls="--", label="Threshold = 3.5")
    ax2.set_ylabel("MAD Anomaly Score")
    ax2.set_xlabel("Stream Steps (Hours)")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_p / "fig04_anomaly_detection_timeline.png")
    plt.close()

    # Figure 5: Concept Drift Detection Timeline
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(steps, drift_scores_c, color="#059669", lw=1.5, label="ADWIN Drift Statistic (Cut Discrepancy)")
    drift_hits = [s for s, d in enumerate(drifts_c) if d["drift_detected"]]
    for idx_dh, dh in enumerate(drift_hits):
        ax.axvline(dh, color="#dc2626", ls="--", lw=1.5, label="Drift Confirmed Alert" if idx_dh == 0 else "")
    ax.set_title("Figure 5: Concept Drift Monitoring & ADWIN Cut-Point Detection", fontweight="bold")
    ax.set_xlabel("Stream Window (Hours)")
    ax.set_ylabel("Drift Score |mu0 - mu1|")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_p / "fig05_concept_drift_timeline.png")
    plt.close()

    # Figure 6: Model Switching and Retraining Timeline
    fig, ax = plt.subplots(figsize=(12, 4))
    model_map = {"lstm": 2, "sarima": 1, "lightweight": 0, "adapted_regime": 3}
    model_y = [model_map.get(m, 2) for m in active_models_c]
    ax.step(steps, model_y, color="#2563eb", lw=2.0, where="post", label="Active Forecaster")
    for idx_re, r_ev in enumerate(retrain_events):
        step_idx = r_ev["metrics"]["step"]
        ax.scatter([step_idx], [model_map.get(r_ev["new_model"], 3)], color="#ef4444", s=100, marker="*", zorder=6, label="Drift Retrain Event" if idx_re == 0 else "")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["Lightweight (Ridge)", "SARIMA (1,1,1)", "PyTorch LSTM", "Adapted Regime"])
    ax.set_title("Figure 6: Dynamic Model Selection & Online Retraining Timeline", fontweight="bold")
    ax.set_xlabel("Stream Steps (Hours)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_p / "fig06_model_switching_timeline.png")
    plt.close()

    # Figure 7: MAE, RMSE, and MAPE Comparison Bar Chart
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
    cfgs = ["Config A\n(SARIMA)", "Config B\n(LSTM)", "Config C\n(Adaptive)"]
    palette = ["#94a3b8", "#38bdf8", "#22c55e"]

    maes = [metrics_a["overall"]["mae"], metrics_b["overall"]["mae"], metrics_c["overall"]["mae"]]
    ax1.bar(cfgs, maes, color=palette)
    ax1.set_title("Overall MAE (kW)", fontweight="bold")
    ax1.set_ylabel("kW")
    for i, v in enumerate(maes): ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    rmses = [metrics_a["overall"]["rmse"], metrics_b["overall"]["rmse"], metrics_c["overall"]["rmse"]]
    ax2.bar(cfgs, rmses, color=palette)
    ax2.set_title("Overall RMSE (kW)", fontweight="bold")
    for i, v in enumerate(rmses): ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    mapes = [metrics_a["overall"]["mape"], metrics_b["overall"]["mape"], metrics_c["overall"]["mape"]]
    ax3.bar(cfgs, mapes, color=palette)
    ax3.set_title("Overall MAPE (%)", fontweight="bold")
    ax3.set_ylabel("%")
    for i, v in enumerate(mapes): ax3.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    plt.suptitle("Figure 7: Forecasting Accuracy Benchmark (A/B/C Configurations)", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(figures_p / "fig07_accuracy_metrics_comparison.png")
    plt.close()

    # Figure 8: Precision, Recall, F1 Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    anom_bar_data = [anomaly_eval["precision"], anomaly_eval["recall"], anomaly_eval["f1_score"]]
    metric_names = ["Precision", "Recall", "F1-Score"]
    bars = ax.bar(metric_names, anom_bar_data, color=["#6366f1", "#8b5cf6", "#ec4899"], width=0.5)
    ax.set_ylim(0, 1.15)
    ax.set_title("Figure 8: Streaming Anomaly Detection Efficacy Metrics", fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)")
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.03, f"{h:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_p / "fig08_detection_efficacy_comparison.png")
    plt.close()

    # Figure 9: Throughput and Latency Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    tputs = [sys_perf["config_A"]["throughput_rps"], sys_perf["config_B"]["throughput_rps"], sys_perf["config_C"]["throughput_rps"]]
    ax1.bar(cfgs, tputs, color=["#cbd5e1", "#93c5fd", "#86efac"], width=0.5)
    ax1.set_title("Pipeline Throughput (rec/s)", fontweight="bold")
    ax1.set_ylabel("Records / Second")
    for i, v in enumerate(tputs): ax1.text(i, v + 2, f"{v:,}", ha="center", fontweight="bold")

    p50s = [sys_perf["config_A"]["p50_latency_ms"], sys_perf["config_B"]["p50_latency_ms"], sys_perf["config_C"]["p50_latency_ms"]]
    p95s = [sys_perf["config_A"]["p95_latency_ms"], sys_perf["config_B"]["p95_latency_ms"], sys_perf["config_C"]["p95_latency_ms"]]
    x = np.arange(len(cfgs))
    ax2.bar(x - 0.15, p50s, 0.3, label="P50 Latency (ms)", color="#3b82f6")
    ax2.bar(x + 0.15, p95s, 0.3, label="P95 Latency (ms)", color="#f97316")
    ax2.set_title("Inference Latency Percentiles (ms)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cfgs)
    ax2.set_ylabel("Milliseconds (ms)")
    ax2.legend()
    for i in range(len(cfgs)):
        ax2.text(x[i] - 0.15, p50s[i] + 0.2, f"{p50s[i]:.1f}", ha="center", fontsize=9)
        ax2.text(x[i] + 0.15, p95s[i] + 0.2, f"{p95s[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
    plt.suptitle("Figure 9: System Telemetry & Computational Performance", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(figures_p / "fig09_throughput_latency_comparison.png")
    plt.close()

    # Figure 10: Process CPU and Memory Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    cpus = [sys_perf["config_A"]["cpu_mean_pct"], sys_perf["config_B"]["cpu_mean_pct"], sys_perf["config_C"]["cpu_mean_pct"]]
    ax1.bar(cfgs, cpus, color=["#cbd5e1", "#fca5a5", "#86efac"], width=0.5)
    ax1.set_title("Measured Process CPU Utilization (%)", fontweight="bold")
    ax1.set_ylabel("CPU %")
    for i, v in enumerate(cpus): ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    mems = [sys_perf["config_A"]["mem_mib"], sys_perf["config_B"]["mem_mib"], sys_perf["config_C"]["mem_mib"]]
    ax2.bar(cfgs, mems, color=["#cbd5e1", "#fca5a5", "#86efac"], width=0.5)
    ax2.set_title("Measured Process Memory (MiB)", fontweight="bold")
    ax2.set_ylabel("Memory (MiB)")
    for i, v in enumerate(mems): ax2.text(i, v + 5, f"{v:.1f}", ha="center", fontweight="bold")

    plt.suptitle("Figure 10: Resource Footprint Across Configurations", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(figures_p / "fig10_cpu_memory_comparison.png")
    plt.close()

    logger.info("All 10 publication figures successfully rendered and saved in runs/figures/")
    return results


def main(argv=None) -> int:
    results = run_evaluation()
    print("\n================ GRAND EVALUATION COMPLETED ================")
    print(f"Evaluated: {results['dataset_rows_evaluated']:,} hours")
    print(f"Config A (SARIMA)  Overall MAE: {results['accuracy']['config_A_fixed_sarima']['overall']['mae']} | Post-Drift MAE: {results['accuracy']['config_A_fixed_sarima']['post_drift']['mae']}")
    print(f"Config B (LSTM)    Overall MAE: {results['accuracy']['config_B_fixed_lstm']['overall']['mae']} | Post-Drift MAE: {results['accuracy']['config_B_fixed_lstm']['post_drift']['mae']}")
    print(f"Config C (Adapt)   Overall MAE: {results['accuracy']['config_C_full_adaptive']['overall']['mae']} | Post-Drift MAE: {results['accuracy']['config_C_full_adaptive']['post_drift']['mae']}")
    print(f"Anomaly Detection  F1-Score: {results['anomaly_detection']['f1_score']} (Precision: {results['anomaly_detection']['precision']}, Recall: {results['anomaly_detection']['recall']})")
    print(f"Concept Drift      Delay Steps: {results['concept_drift_detection']['detection_delay_steps']}")
    print(f"Model Adaptations: {len(results['model_adaptations'])} events recorded")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
