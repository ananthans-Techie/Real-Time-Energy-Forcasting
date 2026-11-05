"""Tests for SyntheticFaultInjector and AnomalyEvaluator."""
import numpy as np
import pandas as pd
import pytest

from cap_anomaly import AnomalyEvaluator, SyntheticFaultInjector
from capstone import constants as C
from capstone.preprocess import aggregate_hourly, build_minute_table
from capstone.synthetic import make_synthetic


def test_synthetic_fault_injector_restricts_to_test_split(hourly):
    injector = SyntheticFaultInjector(seed=99)
    injected_df, labels_df = injector.inject_benchmark_suite(hourly)

    assert len(injected_df) == len(hourly)
    assert not labels_df.empty

    # Verify all labels have event_time >= 2010-01-01
    val_end = pd.Timestamp(C.SPLIT_VAL_END)
    assert (labels_df["event_time"] >= val_end).all()

    # Verify expected labels exist
    labels_set = set(labels_df["label"].unique())
    assert {"spike", "dropout", "missing", "level_shift", "pattern_shift"}.issubset(labels_set)

    # Verify required sidecar columns
    assert set(labels_df.columns) >= {"event_time", "label", "magnitude", "segment_id"}


def test_evaluator_anomaly_metrics():
    # Construct synthetic ground truth and detected events
    timestamps = pd.date_range("2010-01-01 00:00:00", periods=20, freq="1h", tz="UTC")
    gt = pd.DataFrame({
        "event_time": timestamps[:5],
        "label": ["spike", "spike", "dropout", "drop", "missing"],
        "segment_id": ["s1", "s2", "s3", "s4", "s5"],
    })

    # Detections: 4 true positives, 1 false positive, 1 false negative
    detected = [
        {"event_time": timestamps[0], "is_anomaly": True},
        {"event_time": timestamps[1], "is_anomaly": True},
        {"event_time": timestamps[2], "is_anomaly": True},
        {"event_time": timestamps[3], "is_anomaly": True},
        {"event_time": timestamps[10], "is_anomaly": True},  # FP
        {"event_time": timestamps[4], "is_anomaly": False},  # FN
    ]

    metrics = AnomalyEvaluator.evaluate_anomalies(gt, detected)
    assert metrics["true_positives"] == 4
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert round(metrics["precision"], 2) == 0.80
    assert round(metrics["recall"], 2) == 0.80
    assert round(metrics["f1_score"], 2) == 0.80


def test_evaluator_drift_metrics():
    timestamps = pd.date_range("2010-01-01 00:00:00", periods=50, freq="1h", tz="UTC")
    # Shift between index 10 and 30
    gt = pd.DataFrame({
        "event_time": timestamps[10:30],
        "label": ["level_shift"] * 20,
        "segment_id": ["shift_1"] * 20,
    })

    # Drift detected at index 14 (delay = 4 steps)
    detected = [
        {"event_time": timestamps[14], "drift_detected": True},
        {"event_time": timestamps[45], "drift_detected": True},  # false alarm outside shift
    ]

    metrics = AnomalyEvaluator.evaluate_drift(gt, detected, stream_event_times=timestamps)
    assert metrics["total_regime_shifts"] == 1
    assert metrics["detected_shifts"] == 1
    assert metrics["detection_delay_steps"] == 4
    assert metrics["false_alarms_count"] == 1


def test_evaluator_markdown_report_generation():
    anom_metrics = {
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90,
        "detection_delay_steps": 0.0,
        "true_positives": 46,
        "false_positives": 4,
        "false_negatives": 6,
    }
    drift_metrics = {
        "total_regime_shifts": 2,
        "detected_shifts": 2,
        "detection_delay_steps": 12.5,
        "false_alarm_rate": 0.002,
        "false_alarms_count": 1,
    }

    report = AnomalyEvaluator.generate_markdown_report(anom_metrics, drift_metrics)
    assert "# Workstream 3: Anomaly Detection & Concept Drift Evaluation Report" in report
    assert "Precision" in report
    assert "0.9200" in report
    assert "ADWIN" in report
