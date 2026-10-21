"""cap-monitor: sample Kafka consumer lag and per-container CPU / memory during a run.

Each sample is (a) appended to runs/<run_id>/monitor.csv for the report and
(b) published to Kafka topic energy.metrics -> Telegraf -> InfluxDB -> Grafana "Pipeline health".

  cap-monitor --run-id A_ramp --groups telegraf-base,adaptive-processor --interval 2

CPU is `docker stats` CPU% (100% == one full core; can exceed 100 on multi-core).
Needs the docker CLI on the host for CPU/memory; lag only needs Kafka (use --no-docker).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import subprocess
import time
from pathlib import Path

from . import constants as C
from .env import kafka_bootstrap, load_env
from .metrics import MetricsPublisher, metric_event

_UNITS = {"b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3}


def parse_mem_mib(text: str) -> float:
    """'123.4MiB / 7.7GiB' -> 123.4 (MiB of the *used* part)."""
    used = text.split("/")[0].strip()
    m = re.match(r"([\d.]+)\s*([A-Za-z]+)", used)
    if not m:
        return float("nan")
    return float(m.group(1)) * _UNITS.get(m.group(2).lower(), 1) / (1024 ** 2)


def parse_percent(text: str) -> float:
    return float(text.strip().rstrip("%") or "nan")


def docker_samples(name_filter: str) -> list[dict]:
    try:
        out = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}"],
            stderr=subprocess.DEVNULL, text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for line in out.splitlines():
        d = json.loads(line)
        if name_filter in d["Name"]:
            short = d["Name"].replace(name_filter + "-", "", 1).rsplit("-", 1)[0]
            rows.append({"instance": short, "cpu": parse_percent(d["CPUPerc"]),
                         "mem_mib": parse_mem_mib(d["MemUsage"])})
    return rows


class LagReader:
    def __init__(self, bootstrap: str):
        from confluent_kafka import Consumer
        from confluent_kafka.admin import AdminClient
        self.admin = AdminClient({"bootstrap.servers": bootstrap})
        self.probe = Consumer({"bootstrap.servers": bootstrap, "group.id": "cap-monitor-watermarks",
                               "enable.auto.commit": False})

    def lag(self, group: str, topic: str) -> dict:
        """-> {'lag': total records behind, 'end': total log-end offset, 'committed': total}"""
        from confluent_kafka import TopicPartition
        from confluent_kafka.admin import ConsumerGroupTopicPartitions

        md = self.admin.list_topics(topic, timeout=10).topics.get(topic)
        if md is None or not md.partitions:
            return {"lag": 0, "end": 0, "committed": 0}
        tps = [TopicPartition(topic, p) for p in md.partitions]
        res = self.admin.list_consumer_group_offsets([ConsumerGroupTopicPartitions(group, tps)])[group].result()
        committed = {tp.partition: tp.offset for tp in res.topic_partitions}
        lag = end = com = 0
        for p in md.partitions:
            low, high = self.probe.get_watermark_offsets(TopicPartition(topic, p), timeout=10)
            c = committed.get(p, -1)
            c = low if c < 0 else c
            lag += max(high - c, 0); end += high; com += c
        return {"lag": lag, "end": end, "committed": com}

    def close(self):
        self.probe.close()


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--bootstrap", default=kafka_bootstrap())
    ap.add_argument("--groups", default=f"{C.GROUP_BASE},{C.GROUP_ADAPTIVE}")
    ap.add_argument("--topic", default=C.TOPIC_RAW)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--docker-filter", default="energy-pipeline", help="compose project name prefix")
    ap.add_argument("--no-docker", action="store_true")
    ap.add_argument("--output-dir", default="runs")
    a = ap.parse_args(argv)

    from confluent_kafka import Producer
    prod = Producer({"bootstrap.servers": a.bootstrap, "linger.ms": 20})
    pub = MetricsPublisher(prod, C.TOPIC_METRICS)
    reader = LagReader(a.bootstrap)
    out = Path(a.output_dir) / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    stop = {"flag": False}
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.update(flag=True))

    with open(out / "monitor.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric_time_ms", "component", "name", "instance", "value"])
        print(f"[monitor] run={a.run_id} groups={a.groups} interval={a.interval}s -> {out / 'monitor.csv'}")
        while not stop["flag"]:
            t0 = time.time()
            events = []
            for g in [x for x in a.groups.split(",") if x]:
                try:
                    r = reader.lag(g, a.topic)
                except Exception as e:  # noqa: BLE001  (group may not exist yet)
                    continue
                events += [("kafka", "kafka.consumer_lag", g, r["lag"]),
                           ("kafka", "kafka.committed_offset", g, r["committed"]),
                           ("kafka", "kafka.log_end_offset", a.topic, r["end"])]
            if not a.no_docker:
                for d in docker_samples(a.docker_filter):
                    events += [("docker", "docker.cpu_percent", d["instance"], d["cpu"]),
                               ("docker", "docker.mem_mib", d["instance"], d["mem_mib"])]
            ts = int(time.time() * 1000)
            for comp, name, inst, val in events:
                ev = pub.publish(a.run_id, comp, name, val, inst, ts)
                w.writerow([ts, comp, name, inst, val])
            fh.flush(); prod.poll(0)
            time.sleep(max(0.0, a.interval - (time.time() - t0)))
    prod.flush(5); reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
