"""Unit and integration tests for DynamicModelSelector and ModelPerformance tracker."""
import pytest
from capstone import events as E
from capstone.forecasting.selector import DynamicModelSelector, ModelPerformance


def test_performance_tracker_metrics():
    tracker = ModelPerformance()
    # Record predictions for 5 timestamps
    times = [f"2010-01-01T{h:02d}:00:00Z" for h in range(5)]
    preds = [1.0, 1.2, 1.5, 2.0, 1.8]
    actuals = [1.1, 1.0, 1.6, 2.2, 1.7]

    for t, p in zip(times, preds):
        tracker.record_prediction(t, p, latency_ms=10.0)

    for t, a in zip(times, actuals):
        tracker.record_actual(t, a)

    # Errors: |1.0 - 1.1| = 0.1, |1.2 - 1.0| = 0.2, |1.5 - 1.6| = 0.1, |2.0 - 2.2| = 0.2, |1.8 - 1.7| = 0.1
    # MAE = (0.1 + 0.2 + 0.1 + 0.2 + 0.1) / 5 = 0.14
    expected_mae = (0.1 + 0.2 + 0.1 + 0.2 + 0.1) / 5.0
    assert abs(tracker.rolling_mae(window=5) - expected_mae) < 1e-5
    assert tracker.avg_latency_ms() == 10.0


def test_selector_utility_and_normal_selection():
    selector = DynamicModelSelector(
        candidate_models=["lstm", "sarima", "lightweight"],
        default_model="lstm",
        w_acc=0.6,
        w_lat=0.25,
        w_cost=0.15,
    )

    # Simulate performance where LSTM has high accuracy (MAE=0.10)
    for h in range(10):
        t = f"2010-01-01T{h:02d}:00:00Z"
        selector.trackers["lstm"].record_prediction(t, 1.0, latency_ms=5.0)
        selector.trackers["lstm"].record_actual(t, 1.1)

        selector.trackers["sarima"].record_prediction(t, 1.0, latency_ms=25.0)
        selector.trackers["sarima"].record_actual(t, 1.35)

    u_lstm = selector.compute_utility("lstm")
    u_sarima = selector.compute_utility("sarima")
    assert u_lstm > u_sarima

    best_m, switched, trigger, _, _ = selector.evaluate_selection("2010-01-01T11:00:00Z")
    assert best_m == "lstm"
    assert not switched  # Already on LSTM


def test_selector_switches_on_latency_spike():
    selector = DynamicModelSelector(
        candidate_models=["lstm", "sarima", "lightweight"],
        default_model="lstm",
        latency_budget_ms=50.0,
    )

    # Observed latency 120ms exceeds budget of 50ms
    new_model, switched, trigger, val, _ = selector.evaluate_selection(
        event_time_iso="2010-01-01T05:00:00Z",
        observed_latency_ms=120.0,
    )
    assert switched is True
    assert trigger == "latency_spike"
    assert val == 120.0
    assert new_model in ["sarima", "lightweight"]
    assert selector.active_model == new_model


def test_selector_switches_on_degraded_accuracy():
    selector = DynamicModelSelector(
        candidate_models=["lstm", "sarima"],
        default_model="lstm",
        error_threshold_mae=0.60,
    )

    # Inject severe error into active LSTM
    for h in range(10):
        t = f"2010-01-01T{h:02d}:00:00Z"
        selector.trackers["lstm"].record_prediction(t, 1.0, latency_ms=5.0)
        selector.trackers["lstm"].record_actual(t, 2.0)  # error = 1.0 > 0.60

    new_model, switched, trigger, val, _ = selector.evaluate_selection(
        event_time_iso="2010-01-01T11:00:00Z",
        observed_latency_ms=5.0,
    )
    assert switched is True
    assert trigger == "mae_threshold"
    assert val >= 0.60
    assert new_model == "sarima"


def test_selector_switches_on_drift_signal():
    selector = DynamicModelSelector(
        candidate_models=["lstm", "sarima"],
        default_model="lstm",
    )

    new_model, switched, trigger, val, _ = selector.evaluate_selection(
        event_time_iso="2010-01-02T00:00:00Z",
        drift_signal=True,
    )
    assert switched is True
    assert trigger == "drift_signal"
    assert new_model == "sarima"


def test_selector_hysteresis_dwell_time():
    selector = DynamicModelSelector(
        candidate_models=["lstm", "sarima"],
        default_model="lstm",
        min_dwell_hours=12,
        switch_margin=0.10,
    )

    # Immediately after a switch, hours_since_switch is reset
    selector._apply_switch("sarima", "2010-01-01T00:00:00Z")
    assert selector.hours_since_switch == 0

    # Simulate hour 1: LSTM has slightly better utility, but dwell time not met
    selector.trackers["lstm"].errors.append(0.1)
    selector.trackers["sarima"].errors.append(0.15)

    model, switched, trigger, _, _ = selector.evaluate_selection("2010-01-01T01:00:00Z")
    assert not switched
    assert model == "sarima"  # Remains in dwell period


def test_selector_model_event_generation():
    selector = DynamicModelSelector(candidate_models=["lstm", "sarima"])
    ev = selector.create_model_event(
        run_id="C_adaptive-r1",
        meter_id="uci-001",
        event_time_iso="2010-01-01T05:00:00Z",
        prev_model="lstm",
        new_model="sarima",
        trigger_metric="latency_spike",
        trigger_value=120.0,
        trigger_event_id="uci-001-H20100101T040000Z",
        trigger_replay_time_ms=1000,
    )
    E.validate("model_event", ev)
    assert ev["event_type"] == "model_switch"
    assert ev["previous_model"] == "lstm"
    assert ev["new_model"] == "sarima"
    assert ev["trigger_metric"] == "latency_spike"
