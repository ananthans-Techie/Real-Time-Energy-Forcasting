"""Integration tests for StreamingForecaster and end-to-end simulation."""
import numpy as np
import pandas as pd
import pytest

from capstone import constants as C
from capstone import events as E
from capstone.forecasting.forecaster import StreamingForecaster
from capstone.forecasting.models import LightweightForecaster, SARIMAForecaster
from capstone.forecasting.selector import DynamicModelSelector


def make_hourly_event(index: int, target: float = 1.25, meter_id: str = "uci-001") -> dict:
    t0 = pd.Timestamp("2010-01-01 00:00:00Z") + pd.Timedelta(hours=index)
    t1 = t0 + pd.Timedelta(hours=1)
    t0_str = t0.strftime("%Y-%m-%dT%H:00:00Z")
    t1_str = t1.strftime("%Y-%m-%dT%H:00:00Z")

    return {
        "schema_version": "1.0.0",
        "run_id": "test-sim-r1",
        "event_id": E.hourly_event_id(meter_id, t0_str),
        "meter_id": meter_id,
        "event_time": t0_str,
        "window_end": t1_str,
        "target": float(target),
        "features": {
            "global_reactive_power": 0.15,
            "voltage": 240.5,
            "global_intensity": 5.2,
            "sub_metering_1": 0.0,
            "sub_metering_2": 0.0,
            "sub_metering_3": 5.0,
        },
        "n_valid": 60,
        "n_expected": 60,
        "imputed": False,
        "trigger_event_id": f"{meter_id}-20100101T010500Z",
        "trigger_replay_time_ms": 1792522905000 + index * 3600000,
        "emit_time_ms": 1792522905005 + index * 3600000,
    }


def test_streaming_forecaster_pipeline():
    # Setup fast models for simulation
    m1 = LightweightForecaster(lags=(1, 2, 24))
    # Pre-fit on dummy series
    dummy_df = pd.DataFrame({C.TARGET: np.linspace(1.0, 2.0, 50)})
    m1.fit(dummy_df)

    models = {"sarima": m1, "lightweight": m1}
    selector = DynamicModelSelector(
        candidate_models=["sarima", "lightweight"],
        default_model="sarima",
        min_dwell_hours=2,
    )
    forecaster = StreamingForecaster(models=models, selector=selector)

    predictions = []
    # Feed 10 consecutive hourly events
    for h in range(10):
        ev = make_hourly_event(h, target=1.1 + 0.1 * (h % 3))
        pred_ev, model_ev = forecaster.process_hourly_event(ev)
        assert pred_ev is not None
        E.validate("prediction", pred_ev)
        assert pred_ev["event_time"] == ev["window_end"]
        assert pred_ev["trigger_event_id"] == ev["event_id"]
        assert pred_ev["prediction"] >= 0.0
        predictions.append(pred_ev)

    assert len(predictions) == 10
    assert forecaster.stats["records_consumed"] == 10
    assert forecaster.stats["predictions_emitted"] == 10


def test_streaming_forecaster_triggers_model_switch():
    m1 = LightweightForecaster(lags=(1, 2, 24))
    dummy_df = pd.DataFrame({C.TARGET: np.linspace(1.0, 2.0, 50)})
    m1.fit(dummy_df)

    models = {"sarima": m1, "lightweight": m1}
    selector = DynamicModelSelector(
        candidate_models=["sarima", "lightweight"],
        default_model="sarima",
    )
    forecaster = StreamingForecaster(models=models, selector=selector)

    # Process first event normally
    ev0 = make_hourly_event(0, target=1.2)
    pred_ev0, model_ev0 = forecaster.process_hourly_event(ev0)
    assert model_ev0 is None

    # Inject concept drift signal on second event
    ev1 = make_hourly_event(1, target=3.5)
    pred_ev1, model_ev1 = forecaster.process_hourly_event(ev1, drift_signal=True)

    assert model_ev1 is not None
    E.validate("model_event", model_ev1)
    assert model_ev1["event_type"] == "model_switch"
    assert model_ev1["trigger_metric"] == "drift_signal"
    assert model_ev1["new_model"] == "lightweight"
    assert forecaster.selector.active_model == "lightweight"
    assert forecaster.stats["model_switches"] == 1
