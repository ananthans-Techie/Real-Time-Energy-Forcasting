import json

import pandas as pd

from capstone.archive import tidy, write_partitions
from capstone.benchmark import summarize_run
from capstone.latency_probe import percentile, summarize
from capstone.monitor import parse_mem_mib, parse_percent


def test_percentiles():
    xs = list(range(1, 101))
    assert percentile(xs, 50) == 50.5 and percentile(xs, 99) > 99
    assert summarize([10, 20, 30])["max_ms"] == 30


def test_docker_stat_parsing():
    assert parse_percent("12.34%") == 12.34
    assert abs(parse_mem_mib("512MiB / 7.7GiB") - 512) < 1e-9
    assert abs(parse_mem_mib("1.5GiB / 7.7GiB") - 1536) < 1e-9
    assert abs(parse_mem_mib("1048576B / 1GiB") - 1) < 1e-9


def test_archive_partitions_roundtrip(tmp_path):
    t = pd.date_range("2010-01-01", periods=2880, freq="1min", tz="UTC")      # 2 days
    raw = pd.DataFrame({"_time": t, "meter_id": "uci-001", "seq": range(2880), "replay_time_ms": 1,
                        "is_missing": False, "target": 1.0, "voltage": 240.0})
    df = tidy(raw, "r1")
    assert "global_intensity" in df.columns                                    # absent cols are added
    n = write_partitions(df, tmp_path, "r1")
    files = sorted(tmp_path.rglob("part.parquet"))
    assert n == 2880 and len(files) == 2
    assert "date=2010-01-02" in str(files[1])
    back = pd.concat(pd.read_parquet(f) for f in files)
    assert len(back) == 2880


def test_summarize_run(tmp_path):
    d = tmp_path / "run1"; d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"label": "base", "meters": ["a", "b"], "kafka": {"partitions": 3}}))
    (d / "producer_summary.json").write_text(json.dumps({"records_expected": 10, "records_delivered": 10,
        "delivery_errors": 0, "duration_s": 1, "achieved_rps": 10, "max_schedule_lag_s": 0, "target_rps": 10}))
    (d / "latency_summary.json").write_text(json.dumps({"samples": 3, "p50_ms": 5, "p95_ms": 9, "p99_ms": 9,
        "max_ms": 9, "final_record_visible": True, "drain_time_s": 0.4}))
    pd.DataFrame([
        [1, "kafka", "kafka.consumer_lag", "telegraf-base", 100],
        [2, "kafka", "kafka.consumer_lag", "telegraf-base", 40],
        [1, "docker", "docker.cpu_percent", "telegraf", 50],
        [2, "docker", "docker.cpu_percent", "telegraf", 150],
        [1, "docker", "docker.mem_mib", "telegraf", 80],
    ], columns=["metric_time_ms", "component", "name", "instance", "value"]).to_csv(d / "monitor.csv", index=False)
    s = summarize_run(d)
    assert s["lag_max.telegraf-base"] == 100 and s["lag_final.telegraf-base"] == 40
    assert s["cpu_max.telegraf"] == 150 and s["cpu_mean.telegraf"] == 100
    assert s["partitions"] == 3 and s["meters"] == 2 and s["e2e_p95_ms"] == 9
