"""cap-drift: Concept Drift Monitoring Module using ADWIN over Forecast Errors.

Monitors real-time telemetry and model residuals (|y - y_hat|) for sustained statistical
distribution shifts. Distinguishes point anomalies from structural regime changes.

When concept drift is confirmed, emits an `energy.drift` event. This signal directly
triggers Workstream 2's Dynamic Model Selector (`DynamicModelSelector.evaluate_selection`)
to initiate an emergency model re-evaluation or fallback switch.
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

from capstone import constants as C
from capstone.env import kafka_bootstrap, load_env
from capstone.events import build_drift_event, encode, validate

from .adwin import ADWIN, PageHinkley

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Real-time Concept Drift Monitor across multiple meters."""

    def __init__(
        self,
        method: str = "adwin",
        delta: float = 0.005,
        max_window_size: int = 500,
        min_subwindow_size: int = 5,
        step: int = 2,
    ) -> None:
        self.method = method.lower()
        self.delta = delta
        self.max_window_size = max_window_size
        self.min_subwindow_size = min_subwindow_size
        self.step = step

        # Meter-keyed detectors
        if self.method == "adwin":
            self.detectors: dict[str, Any] = defaultdict(
                lambda: ADWIN(
                    delta=self.delta,
                    max_window_size=self.max_window_size,
                    min_subwindow_size=self.min_subwindow_size,
                    step=self.step,
                )
            )
        elif self.method == "page_hinkley":
            self.detectors = defaultdict(
                lambda: PageHinkley(
                    delta=0.01,
                    threshold=30.0,
                )
            )
        else:
            raise ValueError(f"Unsupported drift method '{method}'. Choose 'adwin' or 'page_hinkley'.")

        # Buffers for joining predictions with actuals: (meter_id, event_time) -> record
        self.pending_predictions: dict[tuple[str, str], dict[str, Any]] = {}
        self.pending_actuals: dict[tuple[str, str], dict[str, Any]] = {}

        # De-duplication set
        self.seen_events: set[str] = set()

        # Operational metrics
        self.stats = {
            "records_processed": 0,
            "errors_evaluated": 0,
            "drifts_detected": 0,
        }

    def process_error(
        self,
        run_id: str,
        meter_id: str,
        event_time_iso: str,
        error_val: float,
        trigger_event_id: str,
        trigger_replay_time_ms: int,
    ) -> dict[str, Any]:
        """Directly evaluate an absolute forecast error value for drift."""
        detector = self.detectors[meter_id]
        detected, score, mu0, mu1 = detector.add_element(float(error_val))

        window_size = getattr(detector, "width", getattr(detector, "n", None))

        ev = build_drift_event(
            run_id=run_id,
            meter_id=meter_id,
            event_time_iso=event_time_iso,
            drift_detected=detected,
            drift_score=score,
            mean_error_before=mu0,
            mean_error_after=mu1,
            trigger_event_id=trigger_event_id,
            trigger_replay_time_ms=trigger_replay_time_ms,
            drift_method=self.method,
            window_size=window_size,
        )
        validate("drift", ev)

        self.stats["errors_evaluated"] += 1
        if detected:
            self.stats["drifts_detected"] += 1
            logger.warning(
                f"Concept Drift CONFIRMED for meter {meter_id} at {event_time_iso} | "
                f"score={score:.3f}, mu_before={mu0:.3f}, mu_after={mu1:.3f}"
            )

        return ev

    def process_prediction(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        """Process an energy.predictions event. If matching actual is available, evaluate error."""
        meter_id = str(record["meter_id"])
        event_time = str(record["event_time"])
        key = (meter_id, event_time)

        if key in self.pending_actuals:
            actual_rec = self.pending_actuals.pop(key)
            actual_val = actual_rec.get("target")
            if actual_val is not None:
                err = abs(float(actual_val) - float(record["prediction"]))
                return self.process_error(
                    run_id=str(record["run_id"]),
                    meter_id=meter_id,
                    event_time_iso=event_time,
                    error_val=err,
                    trigger_event_id=str(record["event_id"]),
                    trigger_replay_time_ms=int(record.get("trigger_replay_time_ms", 0)),
                )
        else:
            self.pending_predictions[key] = dict(record)
            if len(self.pending_predictions) > 1000:
                # Evict oldest
                first_key = next(iter(self.pending_predictions))
                self.pending_predictions.pop(first_key)
        return None

    def process_actual(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        """Process an energy.hourly event. If matching prediction is available, evaluate error."""
        meter_id = str(record["meter_id"])
        event_time = str(record["event_time"])
        key = (meter_id, event_time)

        if key in self.pending_predictions:
            pred_rec = self.pending_predictions.pop(key)
            target_val = record.get("target")
            if target_val is not None:
                err = abs(float(target_val) - float(pred_rec["prediction"]))
                return self.process_error(
                    run_id=str(record["run_id"]),
                    meter_id=meter_id,
                    event_time_iso=event_time,
                    error_val=err,
                    trigger_event_id=str(record["event_id"]),
                    trigger_replay_time_ms=int(record.get("replay_time_ms") or record.get("trigger_replay_time_ms") or 0),
                )
        else:
            self.pending_actuals[key] = dict(record)
            if len(self.pending_actuals) > 1000:
                first_key = next(iter(self.pending_actuals))
                self.pending_actuals.pop(first_key)
        return None

    def run_kafka(
        self,
        bootstrap: str,
        group_id: str = C.GROUP_ADAPTIVE + "-drift",
        predictions_topic: str = C.TOPIC_PREDICTIONS,
        actuals_topic: str = C.TOPIC_HOURLY,
        output_topic: str = C.TOPIC_DRIFT,
        limit: int | None = None,
        poll_timeout_s: float = 1.0,
    ) -> None:
        """Run online streaming drift monitor loop."""
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
        consumer.subscribe([predictions_topic, actuals_topic])
        logger.info(
            f"DriftMonitor subscribed to [{predictions_topic}, {actuals_topic}] -> emitting to {output_topic}"
        )

        running = True

        def sig_handler(sig, frame):
            nonlocal running
            running = False
            logger.info("Termination signal received. Draining DriftMonitor...")

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

                topic = msg.topic()
                record = json.loads(msg.value().decode("utf-8"))

                ev = None
                if topic == predictions_topic:
                    ev = self.process_prediction(record)
                elif topic == actuals_topic:
                    ev = self.process_actual(record)

                if ev is not None:
                    producer.produce(
                        topic=output_topic,
                        key=ev["meter_id"].encode("utf-8"),
                        value=encode(ev),
                    )
                    producer.poll(0)
                    count += 1

                    if count % 50 == 0:
                        logger.info(
                            f"Evaluated {count} matched residuals | Drifts detected: {self.stats['drifts_detected']}"
                        )

                if limit is not None and count >= limit:
                    logger.info(f"Reached evaluation limit of {limit} records.")
                    break
        finally:
            producer.flush(10.0)
            consumer.close()
            logger.info(f"DriftMonitor terminated. Summary stats: {self.stats}")


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", default=kafka_bootstrap())
    ap.add_argument("--group", default=C.GROUP_ADAPTIVE + "-drift")
    ap.add_argument("--pred-topic", default=C.TOPIC_PREDICTIONS)
    ap.add_argument("--actual-topic", default=C.TOPIC_HOURLY)
    ap.add_argument("--output-topic", default=C.TOPIC_DRIFT)
    ap.add_argument("--delta", type=float, default=0.005)
    ap.add_argument("--method", default="adwin", choices=["adwin", "page_hinkley"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    monitor = DriftMonitor(method=a.method, delta=a.delta)
    monitor.run_kafka(
        bootstrap=a.bootstrap,
        group_id=a.group,
        predictions_topic=a.pred_topic,
        actuals_topic=a.actual_topic,
        output_topic=a.output_topic,
        limit=a.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
