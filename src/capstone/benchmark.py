"""cap-benchmark: run controlled scenarios and summarise throughput / latency / lag / CPU / memory.

For every scenario in config/benchmark.yaml:
  1. (re)create topics with the scenario's partition count; optionally restart Telegraf;
  2. start cap-monitor, then cap-replay, then cap-probe (once the run manifest exists);
  3. optionally start background commands (teammates' processors for configuration C);
  4. wait for the replay and for the last record to become visible in InfluxDB;
  5. write runs/<run_id>/summary.json and append a row to runs/summary_all.csv.

  cap-benchmark --config config/benchmark.yaml --only baseline_500
  cap-benchmark --summarize runs/<run_id>        # just (re)summarise an existing run
Keep machine, split and measurement windows identical across A/B/C (see docs/MEASUREMENT_PROTOCOL.md).
"""
from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .env import kafka_bootstrap, load_env


# ----------------------------------------------------------------- summarising
def summarize_run(run_dir: str | Path) -> dict:
    d = Path(run_dir)
    s: dict = {"run_id": d.name}
    man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
    s["label"] = man.get("label"); s["partitions"] = man.get("kafka", {}).get("partitions")
    s["meters"] = len(man.get("meters", [])) or None
    if (d / "producer_summary.json").exists():
        p = json.loads((d / "producer_summary.json").read_text())
        s.update({k: p.get(k) for k in ("records_expected", "records_delivered", "delivery_errors",
                                        "duration_s", "achieved_rps", "max_schedule_lag_s", "target_rps")})
    if (d / "latency_summary.json").exists():
        l = json.loads((d / "latency_summary.json").read_text())
        s.update({"e2e_samples": l.get("samples"), "e2e_p50_ms": l.get("p50_ms"), "e2e_p95_ms": l.get("p95_ms"),
                  "e2e_p99_ms": l.get("p99_ms"), "e2e_max_ms": l.get("max_ms"),
                  "drain_time_s": l.get("drain_time_s"), "all_records_visible": l.get("final_record_visible")})
    if (d / "monitor.csv").exists():
        m = pd.read_csv(d / "monitor.csv")
        lag = m[m["name"] == "kafka.consumer_lag"]
        for g, grp in lag.groupby("instance"):
            s[f"lag_max.{g}"] = float(grp["value"].max()); s[f"lag_mean.{g}"] = round(float(grp["value"].mean()), 1)
            s[f"lag_final.{g}"] = float(grp["value"].iloc[-1])
        for name, key in (("docker.cpu_percent", "cpu"), ("docker.mem_mib", "mem_mib")):
            sub = m[m["name"] == name]
            for inst, grp in sub.groupby("instance"):
                s[f"{key}_mean.{inst}"] = round(float(grp["value"].mean()), 2)
                s[f"{key}_max.{inst}"] = round(float(grp["value"].max()), 2)
    (d / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def append_summary(runs_dir: Path, s: dict, scenario: str) -> None:
    row = {"scenario": scenario, **s}
    f = runs_dir / "summary_all.csv"
    new = pd.DataFrame([row])
    if f.exists():
        new = pd.concat([pd.read_csv(f), new], ignore_index=True)
    new.to_csv(f, index=False)


# ------------------------------------------------------------------ orchestration
def _deep(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def run_scenario(sc: dict, runs_dir: Path, bootstrap: str, restart_telegraf: bool) -> dict:
    from .topics import ensure_topics

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{sc.get('prefix', 'bm')}-{sc['name']}-{stamp}"
    print(f"\n=== scenario {sc['name']} -> {run_id}")
    parts = ensure_topics(bootstrap, int(sc.get("raw_partitions", 3)), 1, recreate=True)
    print("    partitions:", parts)
    if restart_telegraf:
        subprocess.run(["docker", "compose", "restart", "telegraf"], check=False)
        time.sleep(8)

    py = [sys.executable, "-m"]
    groups = ",".join(sc.get("groups", ["telegraf-base"]))
    bg = [subprocess.Popen(shlex.split(c)) for c in sc.get("background_cmds", [])]
    mon = subprocess.Popen(py + ["capstone.monitor", "--run-id", run_id, "--groups", groups,
                                 "--output-dir", str(runs_dir), "--bootstrap", bootstrap])
    rcmd = py + ["capstone.replay_producer", "--run-id", run_id, "--label", str(sc.get("label", "base")),
                 "--split", sc.get("split", "test"), "--meters", str(sc.get("meters", 1)),
                 "--output-dir", str(runs_dir), "--bootstrap", bootstrap, "--mode", sc["profile"]["mode"]]
    if sc["profile"].get("rate_rps"): rcmd += ["--rate", str(sc["profile"]["rate_rps"])]
    if sc["profile"].get("speedup"): rcmd += ["--speedup", str(sc["profile"]["speedup"])]
    if sc.get("limit"): rcmd += ["--limit", str(sc["limit"])]
    cfg_file = runs_dir / f"_{run_id}.replay.yaml"
    cfg_file.write_text(yaml.safe_dump({"profile": sc["profile"]}))
    rcmd += ["--config", str(cfg_file)]
    time.sleep(3)                                  # let the monitor take a pre-run baseline
    producer = subprocess.Popen(rcmd)
    manifest = runs_dir / run_id / "manifest.json"
    for _ in range(100):
        if manifest.exists(): break
        time.sleep(0.2)
    probe = subprocess.Popen(py + ["capstone.latency_probe", "--manifest", str(manifest),
                                   "--bootstrap", bootstrap, "--timeout-s", str(sc.get("probe_timeout_s", 1800))])
    producer.wait()
    probe.wait()
    time.sleep(float(sc.get("settle_s", 10)))
    for p in [mon, *bg]:
        p.terminate()
    for p in [mon, *bg]:
        try: p.wait(15)
        except subprocess.TimeoutExpired: p.kill()
    cfg_file.unlink(missing_ok=True)
    s = summarize_run(runs_dir / run_id)
    append_summary(runs_dir, s, sc["name"])
    print("    summary:", json.dumps({k: s.get(k) for k in ("achieved_rps", "e2e_p50_ms", "e2e_p95_ms", "drain_time_s")}))
    return s


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/benchmark.yaml")
    ap.add_argument("--only", help="comma-separated scenario names")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--bootstrap", default=kafka_bootstrap())
    ap.add_argument("--no-restart-telegraf", action="store_true")
    ap.add_argument("--summarize", help="summarise an existing run directory and exit")
    a = ap.parse_args(argv)

    if a.summarize:
        print(json.dumps(summarize_run(a.summarize), indent=2)); return 0
    cfg = yaml.safe_load(Path(a.config).read_text())
    only = set(a.only.split(",")) if a.only else None
    for sc in cfg["scenarios"]:
        if only and sc["name"] not in only:
            continue
        run_scenario(_deep(cfg.get("common", {}), sc), Path(a.runs_dir), a.bootstrap, not a.no_restart_telegraf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
