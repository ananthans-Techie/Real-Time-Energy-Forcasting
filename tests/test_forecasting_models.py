"""Unit tests for forecasting models (SARIMA, LSTM, Lightweight baseline)."""
import numpy as np
import pandas as pd
import pytest

from capstone import constants as C
from capstone.forecasting.models import (
    LightweightForecaster,
    LSTMForecaster,
    SARIMAForecaster,
)


@pytest.fixture
def sample_hourly_df():
    """Synthetic dataframe conforming to hourly schema for unit testing."""
    n = 100
    times = pd.date_range("2009-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    base = 1.0 + 0.5 * np.sin(np.linspace(0, 10, n))
    data = {
        "event_time": times,
        C.TARGET: base + rng.normal(0, 0.05, n),
        "global_reactive_power": 0.1 + rng.uniform(0, 0.05, n),
        "voltage": 240.0 + rng.normal(0, 1.0, n),
        "global_intensity": 4.0 + rng.normal(0, 0.2, n),
        "sub_metering_1": rng.integers(0, 5, n),
        "sub_metering_2": rng.integers(0, 5, n),
        "sub_metering_3": rng.integers(0, 10, n),
        "n_valid": 60,
        "n_expected": 60,
        "imputed": False,
        "split": "train",
    }
    return pd.DataFrame(data)


def test_sarima_forecaster(sample_hourly_df, tmp_path):
    train_df = sample_hourly_df.iloc[:70]
    val_df = sample_hourly_df.iloc[70:]

    forecaster = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 0, 1, 24), context_window=48)
    diag = forecaster.fit(train_df)
    assert diag["model"] == "sarima"
    assert forecaster.fitted_res is not None

    pred, lat, low, high = forecaster.predict_next(val_df)
    assert isinstance(pred, float)
    assert pred >= 0.0
    assert lat >= 0.0

    # Save and load
    save_path = tmp_path / "sarima.pkl"
    forecaster.save(save_path)
    assert save_path.exists()

    loaded = SARIMAForecaster.load(save_path)
    assert loaded.order == forecaster.order
    assert loaded.seasonal_order == forecaster.seasonal_order


def test_lstm_forecaster(sample_hourly_df, tmp_path):
    train_df = sample_hourly_df.iloc[:70]
    val_df = sample_hourly_df.iloc[70:]

    forecaster = LSTMForecaster(window_size=12, hidden_dim=16, num_layers=1, lr=0.01)
    diag = forecaster.fit(train_df, val_df, epochs=2, batch_size=16)
    assert diag["model"] == "lstm"
    assert diag["epochs"] == 2

    pred, lat, low, high = forecaster.predict_next(val_df)
    assert isinstance(pred, float)
    assert pred >= 0.0
    assert lat >= 0.0

    # Determinism test
    pred2, _, _, _ = forecaster.predict_next(val_df)
    assert abs(pred - pred2) < 1e-6

    # Save and load
    save_path = tmp_path / "lstm.pt"
    forecaster.save(save_path)
    assert save_path.exists()
    assert (tmp_path / "lstm.meta.json").exists()
    assert (tmp_path / "lstm.scaler.pkl").exists()

    loaded = LSTMForecaster.load(save_path, device=forecaster.device)
    loaded_pred, _, _, _ = loaded.predict_next(val_df)
    assert abs(pred - loaded_pred) < 1e-4


def test_lightweight_forecaster(sample_hourly_df, tmp_path):
    train_df = sample_hourly_df.iloc[:70]
    val_df = sample_hourly_df.iloc[70:]

    forecaster = LightweightForecaster(lags=(1, 2, 24))
    diag = forecaster.fit(train_df)
    assert diag["model"] == "lightweight"

    pred, lat, low, high = forecaster.predict_next(val_df)
    assert isinstance(pred, float)
    assert pred >= 0.0
    assert lat >= 0.0

    save_path = tmp_path / "lightweight.pkl"
    forecaster.save(save_path)
    loaded = LightweightForecaster.load(save_path)
    loaded_pred, _, _, _ = loaded.predict_next(val_df)
    assert abs(pred - loaded_pred) < 1e-5
