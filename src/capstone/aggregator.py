"""Reference event-time processor: energy.raw (1-min) -> energy.hourly (1-h).

Implements the frozen window rules (docs/DATA_PREPARATION.md) incrementally:

  * windows are [h, h+1h) on *event_time*, never on arrival time;
  * a window closes when the per-(run, meter) watermark
        max(event_time seen) - ALLOWED_LATENESS_S
    passes its end;
  * events for an already-emitted window are dropped and counted (late_dropped);
  * duplicate event_ids inside an open window are ignored (at-least-once delivery);
  * hours with no valid target minute are emitted with imputed=True (LOCF);
  * every emitted record names the raw event that closed it (trigger_*), so that
    end-to-end latency can be computed for *results*, not just for raw readings.

`HourlyAggregator` has no Kafka dependency (unit-testable); `run()` wires it to Kafka.
Teammates may use this as-is or replace it, as long as energy.hourly keeps the frozen schema.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from . import constants as C
from .events import epoch_s, encode, hourly_event_id, iso_from_epoch
from .env import kafka_bootstrap, load_env


@dataclass
class _Window:
    sums: dict = field(default_factory=lambda: {c: 0.0 for c in C.VALUE_COLUMNS})
    counts: dict = field(default_factory=lambda: {c: 0 for c in C.VALUE_COLUMNS})
    n_valid: int = 0
    ids: set = field(default_factory=set)


@dataclass
class _MeterState:
    next_emit: int | None = None          # epoch seconds of the next window to emit
    max_event: int = 0
    windows: dict = field(default_factory=dict)
    last_vals: dict | None = None


class HourlyAggregator:
    def __init__(self, allowed_lateness_s: int = C.ALLOWED_LATENESS_S,
                 window_s: int = C.FORECAST_INTERVAL_S,
                 clock: Callable[[], float] = time.time):
        self.lateness, self.window_s, self.clock = allowed_lateness_s, window_s, clock
        self._state: dict[tuple[str, str], _MeterState] = {}
        self.late_dropped = 0
        self.duplicates = 0
        self.events_in = 0

    # ------------------------------------------------------------------ public
    def add(self, ev: dict) -> list[dict]:
        """Consume one energy.raw event; return the hourly records it caused to close."""
        self.events_in += 1
        key = (ev["run_id"], ev["meter_id"])
        st = self._state.setdefault(key, _MeterState())
        t = epoch_s(ev["event_time"])
        ws = t - t % self.window_s
        if st.next_emit is None:
            st.next_emit = ws
        if ws < st.next_emit:
            self.late_dropped += 1
            return []
        win = st.windows.setdefault(ws, _Window())
        if ev["event_id"] in win.ids:
            self.duplicates += 1
            return []
        win.ids.add(ev["event_id"])
        if not ev["is_missing"]:
            win.n_valid += 1
            win.sums[C.TARGET] += ev["target"]; win.counts[C.TARGET] += 1
            for k, v in ev.get("features", {}).items():
                win.sums[k] += v; win.counts[k] += 1
        st.max_event = max(st.max_event, t)

        out = []
        while st.next_emit + self.window_s + self.lateness <= st.max_event:
            rec = self._emit(key, st, ev)
            if rec is not None:
                out.append(rec)
        return out

    def flush(self, trigger: dict | None = None) -> list[dict]:
        """Close every open window (end of stream / idle timeout)."""
        out = []
        for key, st in self._state.items():
            if st.next_emit is None or not st.windows:
                continue
            last_ws = max(st.windows)
            trig = trigger or {"event_id": "flush", "replay_time_ms": 0}
            while st.next_emit <= last_ws:
                rec = self._emit(key, st, trig)
                if rec is not None:
                    out.append(rec)
        return out

    # ---------------------------------------------------------------- internals
    def _emit(self, key, st: _MeterState, trigger: dict) -> dict | None:
        ws = st.next_emit
        st.next_emit += self.window_s
        win = st.windows.pop(ws, None)
        vals: dict[str, float] = {}
        for c in C.VALUE_COLUMNS:
            if win is not None and win.counts[c] > 0:
                vals[c] = win.sums[c] / win.counts[c]
            elif st.last_vals is not None:
                vals[c] = st.last_vals[c]                       # LOCF (rule M4)
        if len(vals) < len(C.VALUE_COLUMNS):
            return None                                          # leading gap: nothing to carry
        n_valid = win.n_valid if win is not None else 0
        st.last_vals = dict(vals)
        run_id, meter = key
        start = iso_from_epoch(ws)
        return {
            "schema_version": C.SCHEMA_VERSION,
            "run_id": run_id,
            "event_id": hourly_event_id(meter, start),
            "meter_id": meter,
            "event_time": start,
            "window_end": iso_from_epoch(ws + self.window_s),
            "target": vals[C.TARGET],
            "features": {k: vals[k] for k in C.FEATURES},
            "n_valid": n_valid,
            "n_expected": C.MINUTES_PER_WINDOW,
            "imputed": n_valid == 0,
            "trigger_event_id": trigger["event_id"],
            "trigger_replay_time_ms": int(trigger["replay_time_ms"]),
            "emit_time_ms": int(self.clock() * 1000),
        }


def aggregate_stream(events: Iterable[dict], **kw) -> list[dict]:
    """Convenience for tests/offline use: run a whole iterable through the aggregator."""
    agg = HourlyAggregator(**kw)
    out: list[dict] = []
    last = None
    for ev in events:
        out += agg.add(ev)
        last = ev
    out += agg.flush(last)
    return out


# ------------------------------------------------------------------- Kafka runner
def run(bootstrap: str, in_topic: str, out_topic: str, group: str, idle_flush_s: float,
        commit_every: int) -> None:
    from confluent_kafka import Consumer, Producer

    consumer = Consumer({"bootstrap.servers": bootstrap, "group.id": group,
                         "enable.auto.commit": False, "auto.offset.reset": "earliest"})
    producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 5, "compression.type": "lz4"})
    consumer.subscribe([in_topic])
    agg = HourlyAggregator()
    since_commit, last_msg, last_ev = 0, time.time(), None

    def publish(recs):
        for r in recs:
            producer.produce(out_topic, key=r["meter_id"].encode(), value=encode(r))
        producer.poll(0)

    print(f"[aggregator] {in_topic} -> {out_topic} as group '{group}'  (Ctrl-C to stop)")
    try:
        while True:
            msg = consumer.poll(0.5)
            if msg is None:
                if last_ev is not None and time.time() - last_msg > idle_flush_s:
                    publish(agg.flush(last_ev)); producer.flush(); consumer.commit(asynchronous=False)
                    last_ev = None
                continue
            if msg.error():
                print("[aggregator] consumer error:", msg.error()); continue
            last_msg = time.time()
            ev = json.loads(msg.value())
            last_ev = ev
            publish(agg.add(ev))
            since_commit += 1
            if since_commit >= commit_every:
                producer.flush(); consumer.commit(asynchronous=False); since_commit = 0
    except KeyboardInterrupt:
        pass
    finally:
        publish(agg.flush(last_ev)); producer.flush()
        consumer.close()
        print(f"[aggregator] in={agg.events_in} late_dropped={agg.late_dropped} duplicates={agg.duplicates}")


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", default=kafka_bootstrap())
    ap.add_argument("--in-topic", default=C.TOPIC_RAW)
    ap.add_argument("--out-topic", default=C.TOPIC_HOURLY)
    ap.add_argument("--group", default=C.GROUP_ADAPTIVE)
    ap.add_argument("--idle-flush-s", type=float, default=10.0,
                    help="close all open windows after this much silence (end of a finite replay)")
    ap.add_argument("--commit-every", type=int, default=1000)
    a = ap.parse_args(argv)
    run(a.bootstrap, a.in_topic, a.out_topic, a.group, a.idle_flush_s, a.commit_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
