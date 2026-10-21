"""Tiny .env loader so every CLI picks up the same settings without extra dependencies."""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | os.PathLike = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def influx_settings() -> dict:
    load_env()
    return {
        "url": os.environ.get("INFLUX_URL_HOST", "http://localhost:8086"),
        "token": os.environ.get("INFLUX_TOKEN", "capstone-dev-token-change-me"),
        "org": os.environ.get("INFLUX_ORG", "capstone"),
        "bucket": os.environ.get("INFLUX_BUCKET", "energy"),
    }


def kafka_bootstrap() -> str:
    load_env()
    return os.environ.get("KAFKA_BOOTSTRAP_HOST", "localhost:9092")
