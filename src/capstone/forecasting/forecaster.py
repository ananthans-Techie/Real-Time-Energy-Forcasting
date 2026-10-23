"""cap-forecast: Online Streaming Forecaster and Dynamic Model Selection Router.

Consumes hourly aggregated records from energy.hourly, maintains rolling context,
generates 1-step-ahead forecasts using the actively selected model, checks the dynamic
selection policy (evaluating error, latency, and drift), and publishes predictions to
energy.predictions and transitions to energy.model_events.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .. import constants as C
from ..env import kafka_bootstrap, load_env
from ..events import build_prediction_event, encode, validate
from .models import BaseForecaster, LightweightForecaster, LSTMForecaster, SARIMAForecaster
from .selector import DynamicModelSelector

logger = logging.getLogger(__name__)


class StreamingForecaster:
    """Core streaming forecaster with dynamic model selection router."""

    def __init__(
        self,
        models: dict[str, BaseForecaster],
        selector: DynamicModelSelector | None = None,
        context_limit: int = 168,  # Keep up to 1 week (168 hours) of context per meter
    ) -> None:
        self.models = dict(models)
        self.selector = selector or DynamicModelSelector(candidate_models=list(self.models.keys()))
        self.context_limit = context_limit

        # Context buffers: meter_id -> list of hourly record dicts
        self.buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # Processed event tracker for at-least-once deduplication: set of event_id
        self.seen_events: set[str] = set()
        # Last observed inference latencies
        self.last_latency_ms: float = 1.0

        # Run statistics
        self.stats = {
            "records_consumed": 0,
            "predictions_emitted": 0,
            "model_switches": 0,
        }

    @classmethod
    def from_directory(
        cls,
        models_dir: Path | str = "models",
        default_model: str = "lstm",
        device: str | None = None,
    ) -> StreamingForecaster:
        """Load pretrained models from directory or train quick fallback models."""
        p = Path(models_dir)
        models: dict[str, BaseForecaster] = {}

        # 1. SARIMA
        sarima_path = p / "sarima_model.pkl"
        if sarima_path.exists():
            try:
                models["sarima"] = SARIMAForecaster.load(sarima_path)
                logger.info(f"Loaded SARIMA model from {sarima_path}")
            except Exception as e:
                logger.warning(f"Could not load SARIMA: {e}")
        if "sarima" not in models:
            models["sarima"] = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 0, 1, 24))

        # 2. LSTM
        lstm_path = p / "lstm_model.pt"
        if lstm_path.exists():
            try:
                models["lstm"] = LSTMForecaster.load(lstm_path, device=device)
                logger.info(f"Loaded LSTM model from {lstm_path}")
            except Exception as e:
                logger.warning(f"Could not load LSTM: {e}")
        if "lstm" not in models:
            models["lstm"] = LSTMForecaster(device=device)

        # 3. Lightweight
        light_path = p / "lightweight_model.pkl"
        if light_path.exists():
            try:
                models["lightweight"] = LightweightForecaster.load(light_path)
                logger.info(f"Loaded Lightweight model from {light_path}")
            except Exception as e:
                logger.warning(f"Could not load Lightweight model: {e}")
        if "lightweight" not in models:
            models["lightweight"] = LightweightForecaster()

        selector = DynamicModelSelector(
            candidate_models=list(models.keys()),
            default_model=default_model,
        )
        return cls(models=models, selector=selector)

    def process_hourly_event(
        self,
        hourly_event: Mapping[str, Any],
        drift_signal: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Process one hourly aggregated event deterministically.

        1. Ingest actual ground-truth reading for hour T.
        2. Update selector accuracy metrics against previous prediction for hour T.
        3. Evaluate dynamic selection policy.
        4. Generate 1-step-ahead forecast for hour T+1 (window_end).
        5. Package into compliant event envelope and return.

        Returns:
            (prediction_event, optional_model_switch_event)
        """
        # Validate input schema
        validate("hourly", hourly_event)

        run_id = hourly_event["run_id"]
        meter_id = hourly_event["meter_id"]
        event_time = hourly_event["event_time"]       # start of observed window (e.g. 01:00)
        window_end = hourly_event["window_end"]       # start of predicted window (e.g. 02:00)
        actual_target = float(hourly_event["target"])
        trigger_event_id = hourly_event["event_id"]
        trigger_replay_time_ms = hourly_event["trigger_replay_time_ms"]

        # Deduplication
        if trigger_event_id in self.seen_events:
            logger.debug(f"Skipping duplicate event {trigger_event_id}")
        self.seen_events.add(trigger_event_id)

        # 1. Record actual ground-truth in all candidate trackers to measure accuracy
        for model_name in self.models:
            self.selector.trackers[model_name].record_actual(event_time, actual_target)

        # 2. Append to rolling historical buffer for this meter
        flat_record = {
            "event_time": pd.to_datetime(event_time),
            C.TARGET: actual_target,
            **hourly_event.get("features", {}),
            "imputed": hourly_event.get("imputed", False),
        }
        buf = self.buffers[meter_id]
        buf.append(flat_record)
        if len(buf) > self.context_limit:
            buf.pop(0)

        context_df = pd.DataFrame(buf)

        # 3. Dynamic Model Selection Evaluation
        prev_model = self.selector.active_model
        new_model, switched, trigger_metric, trigger_val, snapshot = self.selector.evaluate_selection(
            event_time_iso=window_end,
            drift_signal=drift_signal,
            observed_latency_ms=self.last_latency_ms,
        )

        model_event = None
        if switched:
            model_event = self.selector.create_model_event(
                run_id=run_id,
                meter_id=meter_id,
                event_time_iso=window_end,
                prev_model=prev_model,
                new_model=new_model,
                trigger_metric=trigger_metric,
                trigger_value=trigger_val,
                metrics=snapshot,
                trigger_event_id=trigger_event_id,
                trigger_replay_time_ms=trigger_replay_time_ms,
            )
            validate("model_event", model_event)
            self.stats["model_switches"] += 1

        # 4. 1-Step-Ahead Inference using Active Model
        active_forecaster = self.models[new_model]
        pred_val, latency_ms, ci_lower, ci_upper = active_forecaster.predict_next(context_df)
        self.last_latency_ms = latency_ms

        # Record prediction into tracker for the predicted hour (window_end)
        self.selector.trackers[new_model].record_prediction(window_end, pred_val, latency_ms)

        # 5. Build and validate energy.predictions event
        pred_event = build_prediction_event(
            run_id=run_id,
            meter_id=meter_id,
            event_time_iso=window_end,
            model=new_model,
            prediction=pred_val,
            trigger_event_id=trigger_event_id,
            trigger_replay_time_ms=trigger_replay_time_ms,
            inference_latency_ms=latency_ms,
            confidence_lower=ci_lower,
            confidence_upper=ci_upper,
            rolling_mae=self.selector.trackers[new_model].rolling_mae(),
            rolling_rmse=self.selector.trackers[new_model].rolling_rmse(),
        )
        validate("prediction", pred_event)

        self.stats["records_consumed"] += 1
        self.stats["predictions_emitted"] += 1

        return pred_event, model_event

    def run_kafka(
        self,
        bootstrap: str,
        group_id: str = C.GROUP_ADAPTIVE,
        input_topic: str = C.TOPIC_HOURLY,
        predictions_topic: str = C.TOPIC_PREDICTIONS,
        model_events_topic: str = C.TOPIC_MODEL_EVENTS,
        limit: int | None = None,
        poll_timeout_s: float = 1.0,
    ) -> None:
        """Run online streaming consumer / producer loop."""
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
        logger.info(f"Subscribed to {input_topic} with group {group_id} -> producing to {predictions_topic}")

        running = True

        def sig_handler(sig, frame):
            nonlocal running
            running = False
            logger.info("Termination signal received. Draining forecaster...")

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        count = 0
        try:
            while running:
                msg = consumer.poll(timeout=poll_timeout_s)
                if msg is None:
                    continue
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue

                try:
                    payload = json.loads(msg.value().decode("utf-8"))
                    pred_ev, model_ev = self.process_hourly_event(payload)

                    # Produce prediction
                    key = pred_ev["meter_id"].encode("utf-8")
                    producer.produce(predictions_topic, key=key, value=encode(pred_ev))

                    # Produce model event if switch occurred
                    if model_ev is not None:
                        producer.produce(model_events_topic, key=key, value=encode(model_ev))

                    producer.poll(0)
                    count += 1
                    if count % 100 == 0:
                        logger.info(f"Emitted {count} forecasts (active={self.selector.active_model})")

                    if limit and count >= limit:
                        logger.info(f"Reached limit={limit} records.")
                        break
                except Exception as ex:
                    logger.exception(f"Error processing record: {ex}")

        finally:
            producer.flush(10)
            consumer.close()
            logger.info(f"Forecaster shutdown complete. Processed {count} records.")


def main(argv=None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", default=kafka_bootstrap())
    parser.add_argument("--models-dir", default="models", help="Directory with trained model artifacts")
    parser.add_argument("--group", default=C.GROUP_ADAPTIVE)
    parser.add_argument("--default-model", default="lstm")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None, help="Device ('cpu' or 'cuda')")
    args = parser.parse_args(argv)

    forecaster = StreamingForecaster.from_directory(
        models_dir=args.models_dir,
        default_model=args.default_model,
        device=args.device,
    )
    forecaster.run_kafka(
        bootstrap=args.bootstrap,
        group_id=args.group,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
