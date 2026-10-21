"""cap-replay: replay the UCI minute table into Kafka (topic energy.raw) as an event stream.

* event_time is PRESERVED from the source; replay_time_ms is stamped at emission.
* Load profiles (profile.mode):
    rate     constant records/s                       (e.g. 500 = the base paper's rate)
    ramp     start_rps, +step_rps every step_every_s up to max_rps
    speedup  event-time compression factor            (1 = real time, 60 = 1 h of data per minute)
    max      no pacing (find the ceiling)
* meters > 1 emits every minute once per virtual meter (uci-001..uci-NNN, identical values).
  UCI is ONE household; without this, keying by meter_id puts everything in one partition and
  the partition / consumer-parallelism experiments would be meaningless.
* sinks: kafka | jsonl (file, no broker needed) | null (generator throughput only)
* every run writes runs/<run_id>/manifest.json and producer_summary.json.

Examples
  cap-replay --run-id A_smoke --split test --limit 2000 --rate 500
  cap-replay --run-id ramp1 --mode ramp --meters 8 --split test
  cap-replay --sink jsonl --limit 100 --run-id dry          # no Kafka needed
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import constants as C
from .env import kafka_bootstrap, load_env
from .events import encode, now_ms
from .metrics import MetricsPublisher

DEFAULTS: dict[str, Any] = {
    "run": {"run_id": None, "label": "baseline"},
    "source": {"path": "data/processed/uci_minute.parquet", "split": "test", "start": None, "end": None},
    "meters": 1,
    "profile": {
        "mode": "rate", "rate_rps": 500, "speedup": 1800,
        "ramp": {"start_rps": 500, "step_rps": 500, "step_every_s": 30, "max_rps": 5000},
    },
    "kafka": {
        "bootstrap": None, "topic": C.TOPIC_RAW, "metrics_topic": C.TOPIC_METRICS,
        "producer": {"acks": "1", "linger.ms": 5, "compression.type": "lz4",
                     "batch.num.messages": 10000, "queue.buffering.max.messages": 500000},
    },
    "sink": "kafka", "output_dir": "runs", "limit": None,
}


# ------------------------------------------------------------------- config
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None, overrides: dict) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path and Path(path).exists():
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    cfg = _merge(cfg, overrides)
    if not cfg["run"]["run_id"]:
        cfg["run"]["run_id"] = "run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if not cfg["kafka"]["bootstrap"]:
        cfg["kafka"]["bootstrap"] = kafka_bootstrap()
    return cfg


# ------------------------------------------------------------------ profiles
class Profile:
    """Maps progress (records emitted / event-time offset) to the wall-clock second at which
    that progress is due.  Emitting ahead of schedule -> sleep; behind -> record schedule lag."""

    def __init__(self, cfg: dict):
        self.mode = cfg["mode"]
        self.rate = float(cfg.get("rate_rps", 500))
        self.speedup = float(cfg.get("speedup", 1))
        r = cfg.get("ramp", {})
        self.r0, self.step = float(r.get("start_rps", 500)), float(r.get("step_rps", 500))
        self.every, self.rmax = float(r.get("step_every_s", 30)), float(r.get("max_rps", 5000))
        if self.mode not in ("rate", "ramp", "speedup", "max"):
            raise ValueError(f"unknown profile mode {self.mode!r}")
        if self.mode == "rate" and self.rate <= 0:
            raise ValueError("rate_rps must be > 0")

    def due_s(self, emitted: int, event_offset_s: float) -> float:
        if self.mode == "max":
            return 0.0
        if self.mode == "rate":
            return emitted / self.rate
        if self.mode == "speedup":
            return event_offset_s / self.speedup
        return self._ramp_time(emitted)

    def _ramp_time(self, n: float) -> float:
        t, i = 0.0, 0
        while True:
            rps = min(self.r0 + i * self.step, self.rmax)
            if rps >= self.rmax:
                return t + n / rps
            cap = rps * self.every
            if n <= cap:
                return t + n / rps
            n -= cap; t += self.every; i += 1

    def target_rps(self, t: float) -> float:
        if self.mode == "rate":
            return self.rate
        if self.mode == "ramp":
            return min(self.r0 + int(t // self.every) * self.step, self.rmax)
        return float("nan")


# -------------------------------------------------------------------- sinks
class NullSink:
    def __init__(self): self.sent = self.delivered = self.errors = 0
    def send(self, key: bytes, value: bytes): self.sent += 1; self.delivered += 1
    def poll(self): pass
    def flush(self): pass
    producer = None


class JsonlSink(NullSink):
    def __init__(self, path: Path):
        super().__init__(); path.parent.mkdir(parents=True, exist_ok=True)
        self.fh, self.path = open(path, "w", encoding="utf-8"), path
    def send(self, key, value): self.fh.write(value.decode() + "\n"); self.sent += 1; self.delivered += 1
    def flush(self): self.fh.flush(); self.fh.close()


class KafkaSink(NullSink):
    def __init__(self, cfg: dict):
        from confluent_kafka import Producer
        super().__init__()
        conf = {"bootstrap.servers": cfg["bootstrap"], **cfg["producer"]}
        self.producer = Producer(conf)
        self.topic = cfg["topic"]
        self.last_error = None

    def _cb(self, err, msg):
        if err is not None:
            self.errors += 1; self.last_error = str(err)
        else:
            self.delivered += 1

    def send(self, key: bytes, value: bytes):
        while True:
            try:
                self.producer.produce(self.topic, key=key, value=value, on_delivery=self._cb)
                break
            except BufferError:            # local queue full: serve callbacks, then retry
                self.producer.poll(0.05)
        self.sent += 1
        if self.sent % 200 == 0:
            self.producer.poll(0)

    def poll(self): self.producer.poll(0)
    def flush(self): self.producer.flush(60)


# -------------------------------------------------------------------- source
def load_source(src: dict, limit: int | None) -> pd.DataFrame:
    df = pd.read_parquet(src["path"])
    if src.get("start") or src.get("end"):
        if src.get("start"):
            df = df[df["event_time"] >= pd.Timestamp(src["start"])]
        if src.get("end"):
            df = df[df["event_time"] < pd.Timestamp(src["end"])]
    elif src.get("split") and src["split"] != "all":
        df = df[df["split"] == src["split"]]
    df = df.sort_values("event_time").reset_index(drop=True)
    if limit:
        df = df.iloc[: int(limit)]
    if df.empty:
        raise SystemExit("No source rows selected -- check source.split/start/end and data/processed/.")
    return df


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ main loop
def run_replay(cfg: dict, sink=None, quiet: bool = False) -> dict:
    run_id = cfg["run"]["run_id"]
    run_dir = Path(cfg["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    df = load_source(cfg["source"], cfg.get("limit"))
    n_min = len(df)
    meters = [C.meter_id(i + 1) for i in range(int(cfg["meters"]))]
    profile = Profile(cfg["profile"])

    if sink is None:
        kind = cfg["sink"]
        sink = (KafkaSink({**cfg["kafka"]}) if kind == "kafka"
                else JsonlSink(run_dir / "events.jsonl") if kind == "jsonl" else NullSink())
    mpub = MetricsPublisher(sink.producer, cfg["kafka"]["metrics_topic"]) if cfg["sink"] == "kafka" else None

    # pre-compute per-minute constants
    times = df["event_time"]
    iso = times.dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    compact = [s.replace("-", "").replace(":", "") for s in iso]
    offset_s = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    target = df[C.TARGET].to_numpy(dtype=float)
    feats = {f: df[f].to_numpy(dtype=float) for f in C.FEATURES}
    is_missing = df["is_missing"].to_numpy(dtype=bool)

    partitions = None
    if cfg["sink"] == "kafka":
        try:
            md = sink.producer.list_topics(cfg["kafka"]["topic"], timeout=10)
            partitions = len(md.topics[cfg["kafka"]["topic"]].partitions)
        except Exception:  # noqa: BLE001
            pass

    manifest = {
        "schema_version": C.SCHEMA_VERSION, "run_id": run_id, "label": cfg["run"]["label"],
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0],
                 "cpu_count": __import__("os").cpu_count()},
        "source": {"path": cfg["source"]["path"], "split": cfg["source"]["split"],
                   "start": iso[0], "end_inclusive": iso[-1], "minutes": n_min},
        "meters": meters, "records_expected": n_min * len(meters),
        "profile": cfg["profile"], "kafka": {**{k: v for k, v in cfg["kafka"].items() if k != "producer"},
                                            "producer": cfg["kafka"]["producer"], "partitions": partitions},
        "sink": cfg["sink"], "status": "running",
    }
    ds = Path(cfg["source"]["path"]).parent / "dataset_manifest.json"
    if ds.exists():
        manifest["dataset"] = json.loads(ds.read_text())["files"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if not quiet:
        print(f"[replay] run_id={run_id} label={cfg['run']['label']} minutes={n_min:,} meters={len(meters)} "
              f"records={n_min * len(meters):,} mode={profile.mode} sink={cfg['sink']}")

    seq = 0
    t0 = time.perf_counter()
    last_rep_t, last_rep_n = 0.0, 0
    max_behind = 0.0
    first_replay_ms = last_replay_ms = None
    run_c = C.SCHEMA_VERSION
    for i in range(n_min):
        due = profile.due_s(seq, float(offset_s[i]))
        now = time.perf_counter() - t0
        if due > now:
            gap = due - now
            if gap > 0.0005:
                time.sleep(gap)
        else:
            max_behind = max(max_behind, now - due)

        miss = bool(is_missing[i])
        if not miss:
            fvals = {f: float(feats[f][i]) for f in C.FEATURES if not np.isnan(feats[f][i])}
            tval = float(target[i])
        for m in meters:
            rt = now_ms()
            ev = {"schema_version": run_c, "run_id": run_id, "event_id": f"{m}-{compact[i]}",
                  "meter_id": m, "event_time": iso[i], "seq": seq, "replay_time_ms": rt,
                  "is_missing": miss}
            if not miss:
                ev["target"] = tval
                ev["features"] = fvals
            sink.send(m.encode(), encode(ev))
            seq += 1
            if first_replay_ms is None:
                first_replay_ms = rt
            last_replay_ms = rt

        el = time.perf_counter() - t0
        if mpub is not None and el - last_rep_t >= 1.0:
            rps = (seq - last_rep_n) / (el - last_rep_t)
            mpub.publish(run_id, "producer", "producer.rate_rps", rps, "all")
            mpub.publish(run_id, "producer", "producer.records_sent", seq, "all")
            last_rep_t, last_rep_n = el, seq
            if not quiet and int(el) % 5 == 0:
                print(f"[replay] t={el:6.1f}s sent={seq:,}/{n_min * len(meters):,} rate={rps:,.0f}/s "
                      f"target={profile.target_rps(el):,.0f}/s behind={max_behind:.2f}s")

    sink.flush()
    dur = time.perf_counter() - t0
    summary = {
        "run_id": run_id, "records_expected": n_min * len(meters), "records_sent": sink.sent,
        "records_delivered": sink.delivered, "delivery_errors": sink.errors,
        "duration_s": round(dur, 3), "achieved_rps": round(sink.delivered / dur, 1) if dur else None,
        "max_schedule_lag_s": round(max_behind, 3), "first_replay_time_ms": first_replay_ms,
        "last_replay_time_ms": last_replay_ms, "profile_mode": profile.mode,
        "target_rps": None if profile.mode in ("speedup", "max", "ramp") else profile.rate,
        "meters": len(meters), "partitions": partitions,
    }
    manifest["status"] = "completed"
    manifest["ended_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "producer_summary.json").write_text(json.dumps(summary, indent=2))
    if not quiet:
        print(f"[replay] done: {json.dumps(summary)}")
    return summary


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/replay.yaml")
    ap.add_argument("--run-id"); ap.add_argument("--label")
    ap.add_argument("--source", help="minute parquet (default data/processed/uci_minute.parquet)")
    ap.add_argument("--split", choices=[*C.SPLITS, "all"])
    ap.add_argument("--start", help="ISO event_time, inclusive (overrides --split)")
    ap.add_argument("--end", help="ISO event_time, exclusive")
    ap.add_argument("--mode", choices=["rate", "ramp", "speedup", "max"])
    ap.add_argument("--rate", type=float, help="records/s (mode=rate)")
    ap.add_argument("--speedup", type=float)
    ap.add_argument("--meters", type=int)
    ap.add_argument("--sink", choices=["kafka", "jsonl", "null"])
    ap.add_argument("--bootstrap"); ap.add_argument("--topic")
    ap.add_argument("--limit", type=int, help="replay only the first N minutes (smoke tests)")
    ap.add_argument("--output-dir")
    a = ap.parse_args(argv)

    o: dict[str, Any] = {}
    if a.run_id: o.setdefault("run", {})["run_id"] = a.run_id
    if a.label: o.setdefault("run", {})["label"] = a.label
    if a.source: o.setdefault("source", {})["path"] = a.source
    if a.split: o.setdefault("source", {})["split"] = a.split
    if a.start: o.setdefault("source", {})["start"] = a.start
    if a.end: o.setdefault("source", {})["end"] = a.end
    if a.mode: o.setdefault("profile", {})["mode"] = a.mode
    if a.rate: o.setdefault("profile", {})["rate_rps"] = a.rate
    if a.speedup: o.setdefault("profile", {})["speedup"] = a.speedup
    if a.meters: o["meters"] = a.meters
    if a.sink: o["sink"] = a.sink
    if a.bootstrap: o.setdefault("kafka", {})["bootstrap"] = a.bootstrap
    if a.topic: o.setdefault("kafka", {})["topic"] = a.topic
    if a.limit: o["limit"] = a.limit
    if a.output_dir: o["output_dir"] = a.output_dir

    summary = run_replay(load_config(a.config, o))
    return 1 if summary["delivery_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
