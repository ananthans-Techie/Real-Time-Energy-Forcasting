"""Pipeline metric events (topic energy.metrics -> Telegraf -> InfluxDB -> Grafana)."""
from __future__ import annotations

from . import constants as C
from .events import encode, now_ms


def metric_event(run_id: str, component: str, name: str, value: float,
                 instance: str = "all", metric_time_ms: int | None = None) -> dict:
    return {
        "schema_version": C.SCHEMA_VERSION,
        "run_id": run_id,
        "metric_time_ms": metric_time_ms if metric_time_ms is not None else now_ms(),
        "component": component,
        "name": name,
        "instance": instance,
        "value": float(value),
    }


class MetricsPublisher:
    """Thin wrapper over a confluent_kafka.Producer (or None -> no-op)."""

    def __init__(self, producer, topic: str = C.TOPIC_METRICS):
        self.producer, self.topic = producer, topic

    def publish(self, run_id: str, component: str, name: str, value: float,
                instance: str = "all", metric_time_ms: int | None = None) -> dict:
        ev = metric_event(run_id, component, name, value, instance, metric_time_ms)
        if self.producer is not None:
            self.producer.produce(self.topic, key=run_id.encode(), value=encode(ev))
            self.producer.poll(0)
        return ev
