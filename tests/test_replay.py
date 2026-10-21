import json
import time

import pytest

from capstone import constants as C
from capstone.events import is_valid
from capstone.preprocess import aggregate_hourly
from capstone.replay_producer import Profile, load_config, run_replay


def test_profile_rate_and_speedup():
    p = Profile({"mode": "rate", "rate_rps": 500})
    assert p.due_s(1000, 0) == 2.0
    s = Profile({"mode": "speedup", "speedup": 60})
    assert s.due_s(0, 3600) == 60.0                  # 1 h of event time per minute
    assert Profile({"mode": "max"}).due_s(10**9, 0) == 0.0


def test_profile_ramp_steps():
    p = Profile({"mode": "ramp", "ramp": {"start_rps": 500, "step_rps": 500, "step_every_s": 30, "max_rps": 2000}})
    assert p.due_s(500 * 30, 0) == pytest.approx(30)              # stage 0 fully consumed
    assert p.due_s(500 * 30 + 1000 * 30, 0) == pytest.approx(60)  # +stage 1 at 1000 rps
    # beyond the cap the rate stays at max_rps
    base = p.due_s((500 + 1000 + 1500) * 30, 0)
    assert base == pytest.approx(90)
    assert p.due_s((500 + 1000 + 1500) * 30 + 2000, 0) == pytest.approx(91)
    assert p.target_rps(0) == 500 and p.target_rps(31) == 1000 and p.target_rps(1e6) == 2000


def _cfg(tmp_path, minute, **over):
    path = tmp_path / "uci_minute.parquet"
    minute.to_parquet(path)
    base = {"run": {"run_id": "t1"}, "source": {"path": str(path), "split": "test"}, "meters": 3,
            "profile": {"mode": "max"}, "sink": "jsonl", "output_dir": str(tmp_path / "runs")}
    base.update(over)
    return load_config(None, base)


def test_jsonl_replay_is_schema_valid_and_preserves_event_time(tmp_path, minute):
    s = run_replay(_cfg(tmp_path, minute), quiet=True)
    test_rows = minute[minute["split"] == "test"]
    assert s["records_sent"] == len(test_rows) * 3 and s["delivery_errors"] == 0
    events = [json.loads(l) for l in (tmp_path / "runs/t1/events.jsonl").read_text().splitlines()]
    assert all(is_valid("raw", e) for e in events)
    assert events[0]["event_time"] == test_rows["event_time"].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert {e["meter_id"] for e in events} == {"uci-001", "uci-002", "uci-003"}
    assert sum(e["is_missing"] for e in events) == int(test_rows["is_missing"].sum()) * 3
    man = json.loads((tmp_path / "runs/t1/manifest.json").read_text())
    assert man["status"] == "completed" and man["records_expected"] == s["records_sent"]


def test_split_selection_and_limit(tmp_path, minute):
    s = run_replay(_cfg(tmp_path, minute, meters=1, limit=500, source={"path": str(tmp_path / "uci_minute.parquet"), "split": "val"}), quiet=True)
    assert s["records_sent"] == 500


def test_rate_pacing_is_accurate(tmp_path, minute):
    cfg = _cfg(tmp_path, minute, meters=1, limit=1200, profile={"mode": "rate", "rate_rps": 600}, sink="null")
    t0 = time.perf_counter(); s = run_replay(cfg, quiet=True); dt = time.perf_counter() - t0
    assert 1.8 < dt < 2.6, dt                                     # 1200 records @ 600/s = 2 s
    assert s["achieved_rps"] == pytest.approx(600, rel=0.15)
