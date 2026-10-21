import numpy as np
import pandas as pd

from capstone import constants as C
from capstone.aggregator import HourlyAggregator, aggregate_stream
from capstone.events import build_raw_event, iso_z, validate


def to_events(minute, run="r1", meter="uci-001", shuffle_within=0, seed=0):
    evs = []
    for i, r in enumerate(minute.itertuples(index=False)):
        iso = iso_z(r.event_time.to_pydatetime())
        if r.is_missing:
            evs.append(build_raw_event(run, meter, iso, i, 1000 + i, None, None))
        else:
            f = {k: getattr(r, k) for k in C.FEATURES}
            evs.append(build_raw_event(run, meter, iso, i, 1000 + i, r.target, {k: v for k, v in f.items() if not np.isnan(v)}))
    if shuffle_within:                      # out-of-order inside each block, but within lateness
        rng = np.random.default_rng(seed)
        for s in range(0, len(evs), shuffle_within):
            blk = evs[s:s + shuffle_within]; rng.shuffle(blk); evs[s:s + shuffle_within] = blk
    return evs


def compare(stream, hourly):
    s = pd.DataFrame(stream)
    assert len(s) == len(hourly)
    assert list(pd.to_datetime(s["event_time"], utc=True)) == list(hourly["event_time"])
    assert list(s["n_valid"]) == list(hourly["n_valid"])
    assert list(s["imputed"]) == list(hourly["imputed"])
    assert np.allclose(s["target"], hourly["target"], rtol=1e-9)
    for f in C.FEATURES:
        assert np.allclose([e[f] for e in s["features"]], hourly[f], rtol=1e-9)


def test_stream_matches_batch(minute, hourly):
    out = aggregate_stream(to_events(minute))
    compare(out, hourly)
    for rec in out[:5] + out[-5:]:
        validate("hourly", rec)


def test_out_of_order_within_lateness_still_matches(minute, hourly):
    # shuffle within 4-minute blocks: always < 5 min lateness -> nothing dropped
    agg = HourlyAggregator()
    out = []
    for ev in to_events(minute, shuffle_within=4):
        out += agg.add(ev)
    out += agg.flush()
    assert agg.late_dropped == 0
    compare(out, hourly)


def test_late_event_is_dropped_and_counted(minute):
    evs = to_events(minute.iloc[:600])               # 10 h
    agg = HourlyAggregator()
    for ev in evs:
        agg.add(ev)
    late = evs[10]                                   # an event from hour 0, long after it closed
    late = {**late, "event_id": late["event_id"] + "x"}
    assert agg.add(late) == [] and agg.late_dropped == 1


def test_duplicates_ignored(minute):
    evs = to_events(minute.iloc[:300])
    out_once = aggregate_stream(evs)
    out_dup = aggregate_stream([e for ev in evs for e in (ev, ev)])   # at-least-once redelivery
    assert [r["target"] for r in out_once] == [r["target"] for r in out_dup]


def test_trigger_fields_present(minute):
    out = aggregate_stream(to_events(minute.iloc[:400]))
    assert all(r["trigger_event_id"] and r["trigger_replay_time_ms"] > 0 for r in out)


def test_independent_state_per_run_and_meter(minute):
    a = to_events(minute.iloc[:300], run="r1", meter="uci-001")
    b = to_events(minute.iloc[:300], run="r2", meter="uci-002")
    inter = [e for pair in zip(a, b) for e in pair]
    out = aggregate_stream(inter)
    assert {(r["run_id"], r["meter_id"]) for r in out} == {("r1", "uci-001"), ("r2", "uci-002")}
