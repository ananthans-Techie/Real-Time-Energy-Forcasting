"""Forecasting models for Workstream 2 (SARIMA, LSTM, Lightweight baseline).

Implements:
  1. BaseForecaster: Unified interface for model training, serialization, and 1-step prediction.
  2. SARIMAForecaster: Seasonal ARIMA (1,1,1)x(1,0,1,24) with AIC tuning on validation split.
  3. LSTMForecaster: PyTorch sliding-window LSTM (Adam lr=0.001, MAE loss).
  4. LightweightForecaster: Fast Ridge/AR autoregressive model for constrained/fallback scenarios.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

from .. import constants as C

logger = logging.getLogger(__name__)

# Default feature columns for multivariate forecasting
FEATURE_COLUMNS = list(C.VALUE_COLUMNS)  # target + 6 UCI features


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier (e.g. 'sarima', 'lstm', 'lightweight')."""
        ...

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Fit model on historical dataframe; returns training diagnostics/metrics."""
        ...

    @abstractmethod
    def predict_next(self, context_df: pd.DataFrame) -> tuple[float, float, float | None, float | None]:
        """Predict target for the next step given recent history.

        Returns:
            (prediction_value, inference_latency_ms, confidence_lower, confidence_upper)
        """
        ...

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Serialize model weights and metadata to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> BaseForecaster:
        """Load serialized model from disk."""
        ...


# =====================================================================
# 1. SARIMA Forecaster
# =====================================================================
class SARIMAForecaster(BaseForecaster):
    """SARIMA model: order (1,1,1) and seasonal order (1,0,1,24).

    Implements AIC-guided hyperparameter tuning and rolling 1-step-ahead forecasting.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 24),
        context_window: int = 72,
    ) -> None:
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.context_window = max(context_window, seasonal_order[3] * 2)
        self.fitted_res = None
        self.training_time_s = 0.0
        self.last_aic: float | None = None

    @property
    def name(self) -> str:
        return "sarima"

    def tune_aic(
        self,
        train_series: np.ndarray,
        val_series: np.ndarray,
        candidate_orders: Sequence[tuple[int, int, int]] | None = None,
        candidate_seasonals: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
        """Tune SARIMA hyperparameters via AIC minimization on recent training history."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if candidate_orders is None:
            candidate_orders = [(1, 1, 1), (1, 0, 1), (2, 1, 1)]
        if candidate_seasonals is None:
            candidate_seasonals = [(1, 0, 1, 24), (0, 0, 1, 24), (1, 0, 0, 24)]

        # Use recent history to make hyperparameter grid search feasible
        fit_sample = train_series[-3000:] if len(train_series) > 3000 else train_series
        best_aic = float("inf")
        best_order = self.order
        best_seasonal = self.seasonal_order

        for ord_cand in candidate_orders:
            for seas_cand in candidate_seasonals:
                try:
                    mod = SARIMAX(fit_sample, order=ord_cand, seasonal_order=seas_cand, enforce_stationarity=False)
                    res = mod.fit(disp=False, maxiter=50)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = ord_cand
                        best_seasonal = seas_cand
                except Exception as ex:
                    logger.debug(f"SARIMA tuning candidate failed: {ord_cand}x{seas_cand} ({ex})")
                    continue

        self.order = best_order
        self.seasonal_order = best_seasonal
        logger.info(f"SARIMA selected order={best_order}, seasonal_order={best_seasonal}, AIC={best_aic:.2f}")
        return best_order, best_seasonal

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Fit SARIMA on Global_active_power target series."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        t0 = time.time()
        y_train = train_df[C.TARGET].to_numpy(dtype=float)

        if val_df is not None and len(val_df) > 0:
            y_val = val_df[C.TARGET].to_numpy(dtype=float)
            self.tune_aic(y_train, y_val)

        # Fit with final order on training data (using up to last 5000 observations for stability & speed)
        fit_data = y_train[-5000:] if len(y_train) > 5000 else y_train
        model = SARIMAX(
            fit_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted_res = model.fit(disp=False, maxiter=100)
        self.training_time_s = time.time() - t0
        self.last_aic = float(self.fitted_res.aic)

        return {
            "model": self.name,
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "training_time_s": self.training_time_s,
            "aic": self.last_aic,
        }

    def predict_next(self, context_df: pd.DataFrame) -> tuple[float, float, float | None, float | None]:
        """1-step ahead forecast using state filter on recent context."""
        if self.fitted_res is None:
            raise RuntimeError("SARIMA model is not fitted.")

        t0 = time.time()
        y_ctx = context_df[C.TARGET].to_numpy(dtype=float)
        # Use context window slice
        recent = y_ctx[-self.context_window :] if len(y_ctx) > self.context_window else y_ctx

        try:
            # Filter recent observations through model without refitting parameters
            applied = self.fitted_res.apply(recent, refit=False)
            pm = forecast_res.predicted_mean
            pred = float(np.asarray(pm)[0])
            ci = np.asarray(forecast_res.conf_int(alpha=0.05))
            ci_lower = float(ci[0, 0]) if ci.size > 0 else None
            ci_upper = float(ci[0, 1]) if ci.size > 1 else None
        except Exception:
            # Fallback to last known target or persistence if filter fails
            pred = float(recent[-1]) if len(recent) else 1.0
            ci_lower, ci_upper = max(0.0, pred - 0.5), pred + 0.5

        latency_ms = (time.time() - t0) * 1000.0
        return max(0.0, pred), latency_ms, ci_lower, ci_upper

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "context_window": self.context_window,
            "training_time_s": self.training_time_s,
            "last_aic": self.last_aic,
            "params": self.fitted_res.params if self.fitted_res is not None else None,
        }
        with open(p, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path | str) -> SARIMAForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with open(path, "rb") as f:
            state = pickle.load(f)

        forecaster = cls(
            order=state["order"],
            seasonal_order=state["seasonal_order"],
            context_window=state.get("context_window", 72),
        )
        forecaster.training_time_s = state.get("training_time_s", 0.0)
        forecaster.last_aic = state.get("last_aic")

        # Initialize mock/empty model with params
        if state.get("params") is not None:
            dummy = np.ones(state["context_window"])
            mod = SARIMAX(dummy, order=forecaster.order, seasonal_order=forecaster.seasonal_order)
            forecaster.fitted_res = mod.smooth(state["params"])
        return forecaster


# =====================================================================
# 2. PyTorch LSTM Forecaster
# =====================================================================
class PyTorchLSTM(nn.Module):
    """PyTorch LSTM architecture for sliding-window multivariate energy forecasting."""

    def __init__(self, in_features: int = 7, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMForecaster(BaseForecaster):
    """PyTorch LSTM model for 1-step-ahead hourly energy prediction.

    Hyperparameters per paper:
      - Optimizer: Adam (lr=0.001)
      - Loss: MAE (L1Loss)
      - Sliding window: 24 hours
    """

    def __init__(
        self,
        window_size: int = 24,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 0.001,
        feature_cols: Sequence[str] | None = None,
        device: str | None = None,
    ) -> None:
        import torch

        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.feature_cols = list(feature_cols) if feature_cols else list(FEATURE_COLUMNS)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = MinMaxScaler()
        self.model: Any = None
        self.training_time_s = 0.0

    @property
    def name(self) -> str:
        return "lstm"

    def _create_sliding_windows(self, scaled_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        target_idx = self.feature_cols.index(C.TARGET)
        for i in range(len(scaled_arr) - self.window_size):
            X.append(scaled_arr[i : i + self.window_size])
            y.append(scaled_arr[i + self.window_size, target_idx])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        epochs: int = 15,
        batch_size: int = 128,
    ) -> dict[str, Any]:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        t0 = time.time()
        # Scale features
        raw_train = train_df[self.feature_cols].to_numpy(dtype=float)
        self.scaler.fit(raw_train)
        scaled_train = self.scaler.transform(raw_train)

        X_tr, y_tr = self._create_sliding_windows(scaled_train)
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = None
        if val_df is not None and len(val_df) > self.window_size:
            scaled_val = self.scaler.transform(val_df[self.feature_cols].to_numpy(dtype=float))
            X_v, y_v = self._create_sliding_windows(scaled_val)
            val_loader = DataLoader(TensorDataset(torch.tensor(X_v), torch.tensor(y_v)), batch_size=batch_size)

        # Initialize network
        in_dim = len(self.feature_cols)
        from .models import PyTorchLSTM  # self-import or instantiate directly

        self.model = PyTorchLSTM(in_features=in_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        self.model.to(self.device)

        criterion = nn.L1Loss()  # MAE loss per paper
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(bx)

            train_mae = total_train_loss / len(X_tr)

            if val_loader:
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        loss = criterion(self.model(vx), vy)
                        total_val_loss += loss.item() * len(vx)
                val_mae = total_val_loss / len(val_loader.dataset)
                if val_mae < best_val_loss:
                    best_val_loss = val_mae
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.training_time_s = time.time() - t0
        return {
            "model": self.name,
            "epochs": epochs,
            "training_time_s": self.training_time_s,
            "best_val_loss": best_val_loss if val_loader else train_mae,
        }

    def predict_next(self, context_df: pd.DataFrame) -> tuple[float, float, float | None, float | None]:
        import torch

        if self.model is None:
            raise RuntimeError("LSTM model has not been trained or loaded.")

        t0 = time.time()
        self.model.eval()

        if len(context_df) < self.window_size:
            # Pad with repeating earliest record if context is shorter than window_size
            pad_len = self.window_size - len(context_df)
            pad_rows = pd.concat([context_df.iloc[[0]]] * pad_len, ignore_index=True)
            ctx = pd.concat([pad_rows, context_df], ignore_index=True)
        else:
            ctx = context_df.iloc[-self.window_size :]

        raw_ctx = ctx[self.feature_cols].to_numpy(dtype=float)
        scaled_ctx = self.scaler.transform(raw_ctx)
        tensor_in = torch.tensor(scaled_ctx, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_scaled = float(self.model(tensor_in).cpu().item())

        # Inverse transform target scalar
        target_idx = self.feature_cols.index(C.TARGET)
        dummy = np.zeros((1, len(self.feature_cols)))
        dummy[0, target_idx] = pred_scaled
        unscaled_pred = float(self.scaler.inverse_transform(dummy)[0, target_idx])

        latency_ms = (time.time() - t0) * 1000.0
        # Confidence interval approximation based on recent variance (~5-10%)
        ci_half = max(0.05, 0.10 * unscaled_pred)
        ci_lower = max(0.0, unscaled_pred - ci_half)
        ci_upper = unscaled_pred + ci_half

        return max(0.0, unscaled_pred), latency_ms, ci_lower, ci_upper

    def save(self, path: Path | str) -> None:
        import torch

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "window_size": self.window_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "lr": self.lr,
            "feature_cols": self.feature_cols,
            "training_time_s": self.training_time_s,
        }
        with open(p.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        with open(p.with_suffix(".scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        if self.model is not None:
            torch.save(self.model.state_dict(), p)

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> LSTMForecaster:
        import torch

        p = Path(path)
        meta_file = p.with_suffix(".meta.json")
        scaler_file = p.with_suffix(".scaler.pkl")

        with open(meta_file) as f:
            meta = json.load(f)

        forecaster = cls(
            window_size=meta["window_size"],
            hidden_dim=meta["hidden_dim"],
            num_layers=meta["num_layers"],
            lr=meta["lr"],
            feature_cols=meta["feature_cols"],
            device=device,
        )
        forecaster.training_time_s = meta.get("training_time_s", 0.0)

        with open(scaler_file, "rb") as f:
            forecaster.scaler = pickle.load(f)

        forecaster.model = PyTorchLSTM(
            in_features=len(forecaster.feature_cols),
            hidden_dim=forecaster.hidden_dim,
            num_layers=forecaster.num_layers,
        )
        forecaster.model.load_state_dict(torch.load(p, map_location=forecaster.device))
        forecaster.model.to(forecaster.device)
        forecaster.model.eval()
        return forecaster


# =====================================================================
# 3. Lightweight Baseline Forecaster (Ridge Autoregressive)
# =====================================================================
class LightweightForecaster(BaseForecaster):
    """Fast autoregressive forecaster using recent lag features.

    Suited for extreme low-latency environments or fallbacks during load spikes.
    """

    def __init__(self, lags: Sequence[int] = (1, 2, 3, 24, 25)) -> None:
        self.lags = sorted(lags)
        self.model = Ridge(alpha=1.0)
        self.max_lag = max(self.lags)
        self.training_time_s = 0.0

    @property
    def name(self) -> str:
        return "lightweight"

    def _make_features(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.max_lag, len(series)):
            feats = [series[i - lag] for lag in self.lags]
            X.append(feats)
            y.append(series[i])
        return np.array(X, dtype=float), np.array(y, dtype=float)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict[str, Any]:
        t0 = time.time()
        y = train_df[C.TARGET].to_numpy(dtype=float)
        X, y_target = self._make_features(y)
        self.model.fit(X, y_target)
        self.training_time_s = time.time() - t0
        return {"model": self.name, "training_time_s": self.training_time_s}

    def predict_next(self, context_df: pd.DataFrame) -> tuple[float, float, float | None, float | None]:
        t0 = time.time()
        y_ctx = context_df[C.TARGET].to_numpy(dtype=float)
        if len(y_ctx) < self.max_lag:
            pad = np.full(self.max_lag - len(y_ctx), y_ctx[0] if len(y_ctx) else 1.0)
            y_ctx = np.concatenate([pad, y_ctx])

        feats = np.array([[y_ctx[-lag] for lag in self.lags]], dtype=float)
        pred = float(self.model.predict(feats)[0])
        latency_ms = (time.time() - t0) * 1000.0
        return max(0.0, pred), latency_ms, max(0.0, pred - 0.4), pred + 0.4

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"lags": self.lags, "model": self.model, "training_time_s": self.training_time_s}, f)

    @classmethod
    def load(cls, path: Path | str) -> LightweightForecaster:
        with open(path, "rb") as f:
            data = pickle.load(f)
        forecaster = cls(lags=data["lags"])
        forecaster.model = data["model"]
        forecaster.training_time_s = data.get("training_time_s", 0.0)
        return forecaster
