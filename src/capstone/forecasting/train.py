"""cap-train: Offline training and benchmark verification pipeline.

Loads data/processed/uci_hourly.parquet, trains SARIMA and PyTorch LSTM according to the
paper's exact specifications, validates against Table 1 benchmarks, and saves model
weights into models/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import constants as C
from .models import BaseForecaster, LightweightForecaster, LSTMForecaster, SARIMAForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base paper benchmark targets (Table 1: Im et al. 2024)
BENCHMARK_TARGETS = {
    "sarima": {"mae": 0.3592, "rmse": 0.5103, "tolerance": 0.05},
    "lstm": {"mae": 0.1084, "rmse": 0.0825, "tolerance": 0.05},  # Scaled targets from paper
}


def load_dataset(parquet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load hourly parquet and return (train_df, val_df, test_df)."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed hourly file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    logger.info(
        f"Loaded {len(df):,} hourly rows -> Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}"
    )
    return train_df, val_df, test_df


def evaluate_forecasts(
    actuals: np.ndarray,
    predictions: np.ndarray,
    imputed_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute MAE and RMSE on all hours and valid (non-imputed) hours."""
    errors = predictions - actuals
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    metrics = {
        "all_hours_mae": float(np.mean(abs_errors)),
        "all_hours_rmse": float(np.sqrt(np.mean(sq_errors))),
    }

    if imputed_mask is not None and (~imputed_mask).sum() > 0:
        valid_errors = errors[~imputed_mask]
        metrics["headline_mae"] = float(np.mean(np.abs(valid_errors)))
        metrics["headline_rmse"] = float(np.sqrt(np.mean(valid_errors ** 2)))
    else:
        metrics["headline_mae"] = metrics["all_hours_mae"]
        metrics["headline_rmse"] = metrics["all_hours_rmse"]

    return metrics


def train_sarima(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    quick: bool = False,
) -> tuple[SARIMAForecaster, dict[str, Any]]:
    logger.info("=== Training SARIMA (1,1,1)x(1,0,1,24) ===")
    forecaster = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 0, 1, 24))

    # Fast fit on training split with AIC check
    t0 = time.time()
    train_slice = train_df.iloc[-3000:] if quick else train_df
    diag = forecaster.fit(train_slice, val_df if not quick else None)

    # 1-step ahead evaluation on test set
    logger.info("Evaluating SARIMA on test split...")
    y_test = test_df[C.TARGET].to_numpy(dtype=float)
    y_test_imputed = test_df["imputed"].to_numpy(dtype=bool)

    applied = forecaster.fitted_res.append(y_test)
    fv = applied.fittedvalues
    preds = np.asarray(fv)[-len(y_test):]
    preds = np.clip(preds, 0.0, None)

    eval_res = evaluate_forecasts(y_test, preds, y_test_imputed)
    total_time = time.time() - t0

    model_path = out_dir / "sarima_model.pkl"
    forecaster.save(model_path)
    logger.info(f"SARIMA saved -> {model_path}")
    logger.info(
        f"SARIMA Test Headline MAE: {eval_res['headline_mae']:.4f} (paper: ~0.3592), "
        f"RMSE: {eval_res['headline_rmse']:.4f} (paper: ~0.5103)"
    )

    result = {
        "model": "sarima",
        "order": forecaster.order,
        "seasonal_order": forecaster.seasonal_order,
        "fit_time_s": diag["training_time_s"],
        "total_eval_time_s": total_time,
        "metrics": eval_res,
        "aic": forecaster.last_aic,
        "artifact_path": str(model_path),
    }
    return forecaster, result


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    epochs: int = 15,
    quick: bool = False,
) -> tuple[LSTMForecaster, dict[str, Any]]:
    logger.info("=== Training PyTorch LSTM (sliding window=24, Adam lr=0.001, MAE loss) ===")
    forecaster = LSTMForecaster(window_size=24, hidden_dim=64, num_layers=2, lr=0.001)

    train_slice = train_df.iloc[-4000:] if quick else train_df
    val_slice = val_df.iloc[-500:] if quick else val_df
    num_epochs = 3 if quick else epochs

    diag = forecaster.fit(train_slice, val_slice, epochs=num_epochs)

    # Evaluate on test set
    logger.info("Evaluating LSTM on test split...")
    val_tail = val_df.iloc[-forecaster.window_size :]
    full_test = pd.concat([val_tail, test_df], ignore_index=True)

    y_test = test_df[C.TARGET].to_numpy(dtype=float)
    y_test_imputed = test_df["imputed"].to_numpy(dtype=bool)

    raw_test = full_test[forecaster.feature_cols].to_numpy(dtype=float)
    scaled_test = forecaster.scaler.transform(raw_test)
    X_te, y_te_scaled = forecaster._create_sliding_windows(scaled_test)

    import torch

    forecaster.model.eval()
    with torch.no_grad():
        in_t = torch.tensor(X_te, dtype=torch.float32).to(forecaster.device)
        preds_scaled = forecaster.model(in_t).cpu().numpy()

    # Inverse transform
    target_idx = forecaster.feature_cols.index(C.TARGET)
    dummy = np.zeros((len(preds_scaled), len(forecaster.feature_cols)))
    dummy[:, target_idx] = preds_scaled
    preds_unscaled = forecaster.scaler.inverse_transform(dummy)[:, target_idx]
    preds_unscaled = np.clip(preds_unscaled, 0.0, None)

    eval_unscaled = evaluate_forecasts(y_test, preds_unscaled, y_test_imputed)
    eval_scaled = evaluate_forecasts(y_te_scaled, preds_scaled, y_test_imputed)

    model_path = out_dir / "lstm_model.pt"
    forecaster.save(model_path)
    logger.info(f"LSTM saved -> {model_path}")
    logger.info(
        f"LSTM Unscaled MAE: {eval_unscaled['headline_mae']:.4f}, RMSE: {eval_unscaled['headline_rmse']:.4f}"
    )
    logger.info(
        f"LSTM Scaled MAE: {eval_scaled['headline_mae']:.4f} (paper: ~0.1084), "
        f"RMSE: {eval_scaled['headline_rmse']:.4f} (paper: ~0.0825)"
    )

    result = {
        "model": "lstm",
        "epochs": num_epochs,
        "training_time_s": diag["training_time_s"],
        "metrics_unscaled": eval_unscaled,
        "metrics_scaled": eval_scaled,
        "artifact_path": str(model_path),
    }
    return forecaster, result


def train_lightweight(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[LightweightForecaster, dict[str, Any]]:
    logger.info("=== Training Lightweight Baseline (Ridge AR) ===")
    forecaster = LightweightForecaster()
    diag = forecaster.fit(train_df)

    # Fast evaluation
    y_test = test_df[C.TARGET].to_numpy(dtype=float)
    y_test_imputed = test_df["imputed"].to_numpy(dtype=bool)

    preds = []
    # Feed context
    context = list(train_df[C.TARGET].iloc[-forecaster.max_lag :])
    for val in y_test:
        feats = np.array([[context[-lag] for lag in forecaster.lags]])
        pred = max(0.0, float(forecaster.model.predict(feats)[0]))
        preds.append(pred)
        context.append(val)

    eval_res = evaluate_forecasts(y_test, np.array(preds), y_test_imputed)
    model_path = out_dir / "lightweight_model.pkl"
    forecaster.save(model_path)

    result = {
        "model": "lightweight",
        "training_time_s": diag["training_time_s"],
        "metrics": eval_res,
        "artifact_path": str(model_path),
    }
    return forecaster, result


def run_training_pipeline(
    data_path: Path,
    out_dir: Path,
    epochs: int = 15,
    quick: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = load_dataset(data_path)

    _, sarima_res = train_sarima(train_df, val_df, test_df, out_dir, quick=quick)
    _, lstm_res = train_lstm(train_df, val_df, test_df, out_dir, epochs=epochs, quick=quick)
    _, light_res = train_lightweight(train_df, test_df, out_dir)

    summary = {
        "schema_version": C.SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(data_path),
        "benchmark_targets": BENCHMARK_TARGETS,
        "results": {
            "sarima": sarima_res,
            "lstm": lstm_res,
            "lightweight": light_res,
        },
        "verification": {
            "sarima_mae_matches": abs(sarima_res["metrics"]["headline_mae"] - BENCHMARK_TARGETS["sarima"]["mae"]) < 0.05,
            "sarima_rmse_matches": abs(sarima_res["metrics"]["headline_rmse"] - BENCHMARK_TARGETS["sarima"]["rmse"]) < 0.05,
            "lstm_scaled_mae_matches": abs(lstm_res["metrics_scaled"]["headline_mae"] - BENCHMARK_TARGETS["lstm"]["mae"]) < 0.06,
            "lstm_scaled_rmse_matches": abs(lstm_res["metrics_scaled"]["headline_rmse"] - BENCHMARK_TARGETS["lstm"]["rmse"]) < 0.05,
        },
    }

    report_path = out_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Benchmark summary saved -> {report_path}")

    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/uci_hourly.parquet", help="Path to hourly parquet")
    parser.add_argument("--out", default="models", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs for LSTM")
    parser.add_argument("--quick", action="store_true", help="Quick mode for CI / unit test speed")
    args = parser.parse_args(argv)

    run_training_pipeline(
        data_path=Path(args.data),
        out_dir=Path(args.out),
        epochs=args.epochs,
        quick=args.quick,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
