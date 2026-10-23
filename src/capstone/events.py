"""Builders and validators for the frozen event schemas (v1.0.0)."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping

from . import constants as C

_KIND_TO_FILE = {
    "raw": "energy_raw.v1.schema.json",
    "hourly": "energy_hourly.v1.schema.json",
    "metric": "pipeline_metric.v1.schema.json",
    "prediction": "energy_predictions.v1.schema.json",
    "predictions": "energy_predictions.v1.schema.json",
    "model_event": "energy_model_events.v1.schema.json",
    "model_events": "energy_model_events.v1.schema.json",
}


# ------------------------------------------------------------------ time / ids
def iso_z(dt: datetime) -> str:
    """datetime -> 'YYYY-MM-DDTHH:MM:SSZ' (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso_z(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def epoch_s(s: str) -> int:
    return int(parse_iso_z(s).timestamp())


def iso_from_epoch(sec: int) -> str:
    return iso_z(datetime.fromtimestamp(sec, tz=timezone.utc))


def compact(iso: str) -> str:
    """'2010-01-01T00:00:00Z' -> '20100101T000000Z'"""
    return iso.replace("-", "").replace(":", "")


def raw_event_id(meter_id: str, event_time_iso: str) -> str:
    return f"{meter_id}-{compact(event_time_iso)}"


def hourly_event_id(meter_id: str, window_start_iso: str) -> str:
    return f"{meter_id}-H{compact(window_start_iso)}"


def prediction_event_id(meter_id: str, pred_event_time_iso: str) -> str:
    return f"{meter_id}-P{compact(pred_event_time_iso)}"


def model_event_id(meter_id: str, event_time_iso: str, seq: int | None = None) -> str:
    suffix = f"-{seq}" if seq is not None else ""
    return f"{meter_id}-ME{compact(event_time_iso)}{suffix}"


def now_ms() -> int:
    return int(time.time() * 1000)


# --------------------------------------------------------------------- builders
def build_raw_event(run_id: str, meter_id: str, event_time_iso: str, seq: int,
                    replay_time_ms: int, target: float | None,
                    features: Mapping[str, float | None] | None) -> dict[str, Any]:
    """Build an energy.raw event.  target=None -> missing reading (keys omitted)."""
    ev: dict[str, Any] = {
        "schema_version": C.SCHEMA_VERSION,
        "run_id": run_id,
        "event_id": raw_event_id(meter_id, event_time_iso),
        "meter_id": meter_id,
        "event_time": event_time_iso,
        "seq": seq,
        "replay_time_ms": replay_time_ms,
        "is_missing": target is None,
    }
    if target is not None:
        ev["target"] = float(target)
        ev["features"] = {k: float(v) for k, v in (features or {}).items() if v is not None}
    return ev


def build_prediction_event(
    run_id: str,
    meter_id: str,
    event_time_iso: str,
    model: str,
    prediction: float,
    trigger_event_id: str,
    trigger_replay_time_ms: int,
    inference_latency_ms: float = 0.0,
    confidence_lower: float | None = None,
    confidence_upper: float | None = None,
    rolling_mae: float | None = None,
    rolling_rmse: float | None = None,
    emit_time_ms: int | None = None,
) -> dict[str, Any]:
    """Build an energy.predictions event (1-step-ahead forecast for hour event_time_iso)."""
    ev: dict[str, Any] = {
        "schema_version": C.SCHEMA_VERSION,
        "run_id": run_id,
        "event_id": prediction_event_id(meter_id, event_time_iso),
        "meter_id": meter_id,
        "event_time": event_time_iso,
        "model": str(model),
        "prediction": max(0.0, float(prediction)),
        "horizon_steps": C.FORECAST_HORIZON_STEPS,
        "inference_latency_ms": max(0.0, float(inference_latency_ms)),
        "trigger_event_id": trigger_event_id,
        "trigger_replay_time_ms": int(trigger_replay_time_ms),
        "emit_time_ms": emit_time_ms if emit_time_ms is not None else now_ms(),
    }
    if confidence_lower is not None:
        ev["confidence_lower"] = float(confidence_lower)
    if confidence_upper is not None:
        ev["confidence_upper"] = float(confidence_upper)
    if rolling_mae is not None:
        ev["rolling_mae"] = float(rolling_mae)
    if rolling_rmse is not None:
        ev["rolling_rmse"] = float(rolling_rmse)
    return ev


def build_model_event(
    run_id: str,
    meter_id: str,
    event_time_iso: str,
    event_type: str,
    new_model: str,
    trigger_metric: str,
    previous_model: str | None = None,
    trigger_value: float | str | None = None,
    reason: str | None = None,
    metrics: dict[str, Any] | None = None,
    trigger_event_id: str | None = None,
    trigger_replay_time_ms: int | None = None,
    emit_time_ms: int | None = None,
    seq: int | None = None,
) -> dict[str, Any]:
    """Build an energy.model_events event."""
    ev: dict[str, Any] = {
        "schema_version": C.SCHEMA_VERSION,
        "run_id": run_id,
        "event_id": model_event_id(meter_id, event_time_iso, seq),
        "meter_id": meter_id,
        "event_time": event_time_iso,
        "event_type": str(event_type),
        "previous_model": str(previous_model) if previous_model is not None else None,
        "new_model": str(new_model),
        "trigger_metric": str(trigger_metric),
        "emit_time_ms": emit_time_ms if emit_time_ms is not None else now_ms(),
    }
    if trigger_value is not None:
        ev["trigger_value"] = float(trigger_value) if isinstance(trigger_value, (int, float)) else str(trigger_value)
    if reason:
        ev["reason"] = str(reason)
    if metrics:
        ev["metrics"] = dict(metrics)
    if trigger_event_id:
        ev["trigger_event_id"] = str(trigger_event_id)
    if trigger_replay_time_ms is not None:
        ev["trigger_replay_time_ms"] = int(trigger_replay_time_ms)
    return ev


def encode(event: Mapping[str, Any]) -> bytes:
    """Canonical wire encoding: compact UTF-8 JSON."""
    return json.dumps(event, separators=(",", ":"), allow_nan=False).encode("utf-8")


# ------------------------------------------------------------------- validation
@lru_cache(maxsize=None)
def _validator(kind: str):
    import jsonschema
    fname = _KIND_TO_FILE[kind]
    schema = json.loads(resources.files("capstone").joinpath("schemas", fname).read_text("utf-8"))
    return jsonschema.Draft202012Validator(schema)


def validate(kind: str, event: Mapping[str, Any]) -> None:
    """Raise jsonschema.ValidationError if the event violates the frozen schema."""
    _validator(kind).validate(event)


def is_valid(kind: str, event: Mapping[str, Any]) -> bool:
    return _validator(kind).is_valid(event)
