"""Tests for Workstream 2 schemas (energy.predictions and energy.model_events).

Verifies contract compliance (§4 and §9 of SCHEMA_CONTRACT_v1.0.0.md):
  - Envelope fields (schema_version, run_id, meter_id, event_id, event_time, trigger_*, emit_time_ms)
  - Predictions payload fields
  - Model events payload fields
  - Rejection of invalid types, negative numbers, missing keys, and unexpected extra properties.
"""
import jsonschema
import pytest

from capstone import events as E


def test_prediction_event_schema_valid():
    ev = E.build_prediction_event(
        run_id="C_adaptive-r1",
        meter_id="uci-001",
        event_time_iso="2010-01-01T02:00:00Z",
        model="lstm",
        prediction=1.452,
        trigger_event_id="uci-001-H20100101T010000Z",
        trigger_replay_time_ms=1792522905110,
        inference_latency_ms=15.2,
        confidence_lower=1.10,
        confidence_upper=1.80,
        rolling_mae=0.1084,
        rolling_rmse=0.0825,
    )
    E.validate("prediction", ev)
    assert E.is_valid("prediction", ev)
    assert ev["schema_version"] == "1.0.0"
    assert ev["event_id"] == "uci-001-P20100101T020000Z"
    assert ev["horizon_steps"] == 1


def test_prediction_event_rejects_negative_prediction():
    ev = E.build_prediction_event(
        run_id="C_adaptive-r1",
        meter_id="uci-001",
        event_time_iso="2010-01-01T02:00:00Z",
        model="sarima",
        prediction=0.5,
        trigger_event_id="uci-001-H20100101T010000Z",
        trigger_replay_time_ms=1792522905110,
    )
    ev["prediction"] = -0.01  # invalidate
    with pytest.raises(jsonschema.ValidationError):
        E.validate("prediction", ev)


def test_prediction_event_rejects_additional_properties():
    ev = E.build_prediction_event(
        run_id="C_adaptive-r1",
        meter_id="uci-001",
        event_time_iso="2010-01-01T02:00:00Z",
        model="lstm",
        prediction=1.2,
        trigger_event_id="uci-001-H20100101T010000Z",
        trigger_replay_time_ms=1792522905110,
    )
    ev["unknown_extra_field"] = "bad"
    with pytest.raises(jsonschema.ValidationError):
        E.validate("prediction", ev)


def test_prediction_event_rejects_invalid_event_time_format():
    ev = E.build_prediction_event(
        run_id="C_adaptive-r1",
        meter_id="uci-001",
        event_time_iso="2010-01-01T02:00:00Z",
        model="lstm",
        prediction=1.2,
        trigger_event_id="uci-001-H20100101T010000Z",
        trigger_replay_time_ms=1792522905110,
    )
    ev["event_time"] = "2010-01-01 02:00:00"  # missing T and Z
    with pytest.raises(jsonschema.ValidationError):
        E.validate("prediction", ev)


def test_model_event_schema_valid():
    ev = E.build_model_event(
        run_id="C_adaptive-r1",
        meter_id="uci-001",
        event_time_iso="2010-01-05T12:00:00Z",
        event_type="model_switch",
        previous_model="lstm",
        new_model="sarima",
        trigger_metric="mae_threshold",
        trigger_value=0.72,
        reason="Active LSTM rolling MAE exceeded 0.65 threshold",
        metrics={"lstm_mae": 0.72, "sarima_mae": 0.38},
        trigger_event_id="uci-001-H20100105T110000Z",
        trigger_replay_time_ms=1792523000000,
    )
    E.validate("model_event", ev)
    assert E.is_valid("model_event", ev)
    assert ev["event_id"] == "uci-001-ME20100105T120000Z"


def test_model_event_initial_selection():
    ev = E.build_model_event(
        run_id="C_adaptive-r1",
        meter_id="uci-001",
        event_time_iso="2010-01-01T00:00:00Z",
        event_type="initial_selection",
        previous_model=None,
        new_model="lstm",
        trigger_metric="initial",
        reason="Pipeline initialized with default high-accuracy model",
    )
    E.validate("model_event", ev)
    assert E.is_valid("model_event", ev)


def test_model_event_rejects_unknown_event_type():
    ev = E.build_model_event(
        run_id="C_adaptive-r1",
        meter_id="uci-001",
        event_time_iso="2010-01-01T00:00:00Z",
        event_type="random_unsupported_action",
        new_model="lstm",
        trigger_metric="initial",
    )
    with pytest.raises(jsonschema.ValidationError):
        E.validate("model_event", ev)
