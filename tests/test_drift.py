"""Unit and integration tests for Workstream 3 Concept Drift Monitoring.

Validates:
  - ADWIN change detection on stationary vs shifted distributions.
  - Page-Hinkley cumulative change detection.
  - DriftMonitor error processing and schema compliance on energy.drift.
  - Stream matching between predictions (energy.predictions) and actuals (energy.hourly).
  - Direct handoff and emergency model re-evaluation trigger in Workstream 2's DynamicModelSelector.
"""
import jsonschema
import numpy as np
import pytest

from cap_drift import ADWIN, DriftMonitor, PageHinkley
from capstone import events as E
from capstone.forecasting.selector import DynamicModelSelector


def test_adwin_stationary_no_false_alarms():
    adwin = ADWIN(delta=0.002, max_window_size=200)
    rng = np.random.default_rng(123)

    drifts = []
    # 80 observations from stationary normal distribution ~ N(0.2, 0.02)
    for _ in range(80):
        val = rng.normal(0.2, 0.02)
        detected, score, mu0, mu1 = adwin.add_element(val)
        if detected:
            drifts.append(detected)

    assert len(drifts) == 0, "ADWIN should not trigger on stationary stream"


def test_adwin_detects_sustained_level_shift():
    adwin = ADWIN(delta=0.005, max_window_size=200, min_subwindow_size=5)
    rng = np.random.default_rng(456)

    # 40 stationary baseline errors ~ 0.15 kW
    for _ in range(40):
        adwin.add_element(float(rng.normal(0.15, 0.02)))

    # Sustained regime shift: error jumps to 1.85 kW
    detected_at_step = None
    for step in range(1, 40):
        val = float(rng.normal(1.85, 0.05))
        detected, score, mu0, mu1 = adwin.add_element(val)
        if detected:
            detected_at_step = step
            assert score >= 1.0
            assert mu1 > mu0
            break

    assert detected_at_step is not None, "ADWIN failed to detect sustained level shift"
    assert detected_at_step <= 25, f"Detection delay too high: {detected_at_step} steps"


def test_page_hinkley_detection():
    ph = PageHinkley(delta=0.01, threshold=20.0, min_instances=10)
    for _ in range(30):
        ph.add_element(0.2)

    detected = False
    for _ in range(30):
        det, score, _, _ = ph.add_element(1.5)
        if det:
            detected = True
            break

    assert detected is True, "PageHinkley failed to detect abrupt change"


def test_drift_monitor_process_error_and_schema():
    monitor = DriftMonitor(method="adwin", delta=0.005)

    ev = monitor.process_error(
        run_id="C_adaptive-r1",
        meter_id="uci-001",
        event_time_iso="2010-01-05T12:00:00Z",
        error_val=0.22,
        trigger_event_id="uci-001-H20100105T120000Z",
        trigger_replay_time_ms=1792500000000,
    )
    assert ev["schema_version"] == "1.0.0"
    assert ev["event_id"] == "uci-001-D20100105T120000Z"
    assert ev["drift_method"] == "adwin"
    E.validate("drift", ev)


def test_drift_monitor_prediction_actual_matching():
    monitor = DriftMonitor(method="adwin")

    # Send prediction first for hour 12:00:00Z
    pred_rec = {
        "run_id": "C_adaptive-r1",
        "event_id": "uci-001-P20100105T120000Z",
        "meter_id": "uci-001",
        "event_time": "2010-01-05T12:00:00Z",
        "prediction": 1.25,
        "trigger_replay_time_ms": 1792500000000,
    }
    res1 = monitor.process_prediction(pred_rec)
    assert res1 is None  # pending actual

    # Send matching actual for hour 12:00:00Z
    actual_rec = {
        "run_id": "C_adaptive-r1",
        "event_id": "uci-001-H20100105T120000Z",
        "meter_id": "uci-001",
        "event_time": "2010-01-05T12:00:00Z",
        "target": 1.30,
        "replay_time_ms": 1792500001000,
    }
    res2 = monitor.process_actual(actual_rec)
    assert res2 is not None
    assert res2["event_time"] == "2010-01-05T12:00:00Z"
    E.validate("drift", res2)


def test_drift_signal_triggers_dynamic_selector():
    """Verify handoff: confirmed drift triggers Workstream 2's DynamicModelSelector."""
    selector = DynamicModelSelector(
        candidate_models=["lstm", "sarima", "lightweight"],
        default_model="lstm",
        min_dwell_hours=10,  # normally would resist switching
    )
    selector.hours_since_switch = 0  # In dwell period
    assert selector.active_model == "lstm"

    # Routine step without drift -> no switch due to dwell time
    new_model, switched, trigger, val, _ = selector.evaluate_selection(
        event_time_iso="2010-01-02T01:00:00Z",
        drift_signal=False,
    )
    assert switched is False
    assert selector.active_model == "lstm"

    # Now emit confirmed drift signal from Workstream 3!
    new_model2, switched2, trigger2, val2, _ = selector.evaluate_selection(
        event_time_iso="2010-01-02T02:00:00Z",
        drift_signal=True,  # Confirmed drift from energy.drift
    )
    assert switched2 is True
    assert trigger2 == "drift_signal"
    assert new_model2 != "lstm"
    assert selector.active_model == new_model2
