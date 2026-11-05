"""cap-anomaly: Real-Time Anomaly Detection using Rolling Median and MAD.

Implements non-parametric anomaly detection over power readings (Global_active_power)
using a rolling window Median Absolute Deviation (MAD). Non-parametric MAD provides
robustness against extreme outliers and transient spikes, ensuring high breakdown points.

Outputs structured events to `energy.anomalies` according to SCHEMA_CONTRACT_v1.0.0.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import defaultdict, deque
from typing import Any, Mapping

import numpy as np

from capstone import constants as C
from capstone.env import kafka_bootstrap, load_env
from capstone.events import build_anomaly_event, encode, validate

logger = logging.getLogger(__name__)


class RollingMADDetector:
    """Non-parametric rolling median and MAD anomaly detector."""

    def __init__(
        self,
        window_size: int = 24,
        threshold: float = 3.5,
        min_samples: int = 5,
        dropout_threshold: float = 0.05,
    ) -> None:
        """Initialize detector.

        Args:
            window_size: Number of past observations in baseline window.
            threshold: Modified Z-score threshold for anomaly classification (standard >= 3.5).
            min_samples: Minimum observations required before flagging anomalies.
            dropout_threshold: Fraction of expected value below which a reading near zero is a dropout.
        """
        self.window_size = max(3, int(window_size))
        self.threshold = float(threshold)
        self.min_samples = max(2, int(min_samples))
        self.dropout_threshold = float(dropout_threshold)

        # Buffer of past non-anomalous / baseline observations
        self.history: deque[float] = deque(maxlen=self.window_size)

    def update(self, value: float | None) -> tuple[float | None, float, bool, str]:
        """Process an observation and evaluate anomaly status.

        Returns:
            (expected_value, anomaly_score, is_anomaly, anomaly_type)
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            # Missing reading anomaly
            median_val = float(np.median(self.history)) if len(self.history) >= self.min_samples else None
            return median_val, 0.0, True, "missing"

        y = float(value)

        # If not enough history, collect baseline and return normal
        if len(self.history) < self.min_samples:
            self.history.append(y)
            return y, 0.0, False, "normal"

        arr = np.array(self.history, dtype=float)
        median_val = float(np.median(arr))
        abs_deviations = np.abs(arr - median_val)
        mad = float(np.median(abs_deviations))

        # Check for zero / near-zero dropout condition
        if y <= 0.001 and median_val > 0.4:
            # Clear dropout anomaly
            # Do not contaminate rolling history with dropout
            return median_val, 99.0, True, "dropout"

        # Modified Z-score calculation (Boris Iglewicz and David Hoaglin 1993)
        # 0.6745 is the 75th percentile of standard normal distribution: E[MAD] = sigma / 1.4826 = 0.6745 * sigma
        if mad > 1e-5:
            score = float(0.6745 * abs(y - median_val) / mad)
        else:
            # If MAD is virtually 0 (constant baseline), check std dev or absolute diff
            std_val = float(np.std(arr))
            if std_val > 1e-5:
                score = float(abs(y - median_val) / std_val)
            else:
                score = float(abs(y - median_val) / 0.1)

        is_anomaly = bool(score >= self.threshold)

        if is_anomaly:
            anomaly_type = "spike" if y > median_val else "drop"
            # Point anomalies are NOT added to the baseline history to prevent masking subsequent anomalies
        else:
            anomaly_type = "normal"
            self.history.append(y)

        return median_val, score, is_anomaly, anomaly_type

    def reset(self) -> None:
        """Clear detector history."""
        self.history.clear()


class AnomalyDetector:
    """Multi-meter streaming anomaly detector module."""

    def __init__(
        self,
        window_size: int = 24,
        threshold: float = 3.5,
        min_samples: int = 5,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples

        # Meter-keyed detectors
        self.detectors: dict[str, RollingMADDetector] = defaultdict(
            lambda: RollingMADDetector(
                window_size=self.window_size,
                threshold=self.threshold,
                min_samples=self.min_samples,
            )
        )
        self.seen_events: set[str] = set()
        self.stats = {
            "records_processed": 0,
            "anomalies_detected": 0,
            "spikes": 0,
            "drops": 0,
            "dropouts": 0,
            "missing": 0,
        }

    def process_record(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        """Process an incoming energy.hourly or energy.raw record.

        Returns an energy.anomalies event dict or None if duplicate.
        """
        event_id = str(record["event_id"])
        if event_id in self.seen_events:
            return None
        self.seen_events.add(event_id)
        if len(self.seen_events) > 50000:
            self.seen_events.clear()

        run_id = str(record["run_id"])
        meter_id = str(record["meter_id"])
        event_time_iso = str(record["event_time"])
        trigger_replay_time_ms = int(record.get("replay_time_ms") or record.get("trigger_replay_time_ms") or 0)

        # Target value handling (None if missing)
        target_val = record.get("target")
        is_missing = bool(record.get("is_missing", False))
        if is_missing or target_val is None:
            val = None
        else:
            val = float(target_val)

        detector = self.detectors[meter_id]
        expected_val, score, is_anomaly, anomaly_type = detector.update(val)

        # Build schema compliant event
        ev = build_anomaly_event(
            run_id=run_id,
            meter_id=meter_id,
            event_time_iso=event_time_iso,
            target_value=val,
            expected_value=expected_val,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            trigger_event_id=event_id,
            trigger_replay_time_ms=trigger_replay_time_ms,
            anomaly_type=anomaly_type,
            threshold=self.threshold,
            window_size=self.window_size,
        )
        validate("anomaly", ev)

        self.stats["records_processed"] += 1
        if is_anomaly:
            self.stats["anomalies_detected"] += 1
            if anomaly_type == "spike":
                self.stats["spikes"] += 1
            elif anomaly_type == "drop":
                self.stats["drops"] += 1
            elif anomaly_type == "dropout":
                self.stats["dropouts"] += 1
            elif anomaly_type == "missing":
                self.stats["missing"] += 1

        return ev

    def run_kafka(
        self,
        bootstrap: str,
        group_id: str = C.GROUP_ADAPTIVE + "-anomaly",
        input_topic: str = C.TOPIC_HOURLY,
        output_topic: str = C.TOPIC_ANOMALIES,
        limit: int | None = None,
        poll_timeout_s: float = 1.0,
    ) -> None:
        """Run real-time streaming anomaly detection consumer/producer loop."""
        from confluent_kafka import Consumer, Producer

        conf_c = {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
        conf_p = {
            "bootstrap.servers": bootstrap,
            "linger.ms": 5,
        }

        consumer = Consumer(conf_c)
        producer = Producer(conf_p)
        consumer.subscribe([input_topic])
        logger.info(f"AnomalyDetector subscribed to {input_topic} (group {group_id}) -> emitting to {output_topic}")

        running = True

        def sig_handler(sig, frame):
            nonlocal running
            running = False
            logger.info("Termination signal received. Draining AnomalyDetector...")

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        count = 0
        try:
            while running:
                msg = consumer.poll(poll_timeout_s)
                if msg is None:
                    continue
                if msg.error():
                    logger.warning(f"Kafka consumer error: {msg.error()}")
                    continue

                raw_val = msg.value().decode("utf-8")
                record = json.loads(raw_val)

                ev = self.process_record(record)
                if ev is not None:
                    producer.produce(
                        topic=output_topic,
                        key=ev["meter_id"].encode("utf-8"),
                        value=encode(ev),
                    )
                    producer.poll(0)
                    count += 1

                    if count % 100 == 0:
                        logger.info(
                            f"Processed {count} records | Detected {self.stats['anomalies_detected']} anomalies "
                            f"(Spikes: {self.stats['spikes']}, Drops: {self.stats['drops']}, Dropouts: {self.stats['dropouts']})"
                        )

                if limit is not None and count >= limit:
                    logger.info(f"Reached processing limit of {limit} records.")
                    break
        finally:
            producer.flush(10.0)
            consumer.close()
            logger.info(f"AnomalyDetector terminated. Summary stats: {self.stats}")


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", default=kafka_bootstrap())
    ap.add_argument("--group", default=C.GROUP_ADAPTIVE + "-anomaly")
    ap.add_argument("--input-topic", default=C.TOPIC_HOURLY)
    ap.add_argument("--output-topic", default=C.TOPIC_ANOMALIES)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=3.5)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    detector = AnomalyDetector(window_size=a.window, threshold=a.threshold)
    detector.run_kafka(
        bootstrap=a.bootstrap,
        group_id=a.group,
        input_topic=a.input_topic,
        output_topic=a.output_topic,
        limit=a.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
