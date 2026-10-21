"""cap-topics: create (or recreate) the frozen Kafka topics.

  cap-topics --raw-partitions 3                 # create if missing
  cap-topics --raw-partitions 6 --recreate      # partition-scaling experiment
Topic auto-creation is disabled in docker-compose.yml on purpose: partition counts are an
experimental variable and must be explicit.
"""
from __future__ import annotations

import argparse
import sys
import time

from . import constants as C
from .env import kafka_bootstrap, load_env


def wait_for_broker(admin, timeout_s: float = 60.0) -> None:
    t0 = time.time()
    while True:
        try:
            admin.list_topics(timeout=3)
            return
        except Exception as e:  # noqa: BLE001
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"Kafka not reachable after {timeout_s:.0f}s: {e}")
            time.sleep(1)


def ensure_topics(bootstrap: str, raw_partitions: int, other_partitions: int,
                  recreate: bool, topics=C.ALL_TOPICS) -> dict[str, int]:
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    wait_for_broker(admin)
    existing = set(admin.list_topics(timeout=10).topics)
    if recreate:
        gone = [t for t in topics if t in existing]
        for t, f in admin.delete_topics(gone, operation_timeout=30).items():
            f.result()
        deadline = time.time() + 30
        while time.time() < deadline and set(admin.list_topics(timeout=10).topics) & set(gone):
            time.sleep(0.5)
        existing -= set(gone)
    todo = [NewTopic(t, num_partitions=raw_partitions if t == C.TOPIC_RAW else other_partitions,
                     replication_factor=1) for t in topics if t not in existing]
    for t, f in admin.create_topics(todo).items():
        f.result()
    md = admin.list_topics(timeout=10).topics
    return {t: len(md[t].partitions) for t in topics if t in md}


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", default=kafka_bootstrap())
    ap.add_argument("--raw-partitions", type=int, default=3)
    ap.add_argument("--other-partitions", type=int, default=1)
    ap.add_argument("--recreate", action="store_true", help="delete + recreate (drops all data)")
    a = ap.parse_args(argv)
    res = ensure_topics(a.bootstrap, a.raw_partitions, a.other_partitions, a.recreate)
    for t, p in res.items():
        print(f"{t:22s} partitions={p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
