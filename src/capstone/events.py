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
