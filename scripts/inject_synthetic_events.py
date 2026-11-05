#!/usr/bin/env python3
"""CLI utility to inject synthetic anomalies and concept drift scenarios.

Usage:
  python scripts/inject_synthetic_events.py --source data/processed/uci_minute.parquet \
                                           --out-data data/processed/injected_minute.parquet \
                                           --out-labels data/processed/labels.parquet
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cap_anomaly.synthetic_injector import main

if __name__ == "__main__":
    sys.exit(main())
