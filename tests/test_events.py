import copy

import jsonschema
import pytest

from capstone import constants as C
from capstone.events import (build_raw_event, encode, hourly_event_id, is_valid, raw_event_id, validate)
from capstone.metrics import metric_event

FEATS = {f: 1.0 for f in C.FEATURES}


def ok_raw():
    return build_raw_event("run-1", "uci-001", "2010-01-01T00:00:00Z", 0, 1_700_000_000_000, 4.2, FEATS)


def test_valid_raw_and_ids():
    ev = ok_raw()
    validate("raw", ev)
    assert ev["event_id"] == "uci-001-20100101T000000Z" == raw_event_id("uci-001", "2010-01-01T00:00:00Z")
    assert hourly_event_id("uci-001", "2010-01-01T01:00:00Z") == "uci-001-H20100101T010000Z"
    assert list(ev)[:4] == ["schema_version", "run_id", "event_id", "meter_id"]


def test_missing_event_omits_keys():
    ev = build_raw_event("run-1", "uci-001", "2010-01-01T00:01:00Z", 1, 1, None, None)
    validate("raw", ev)
    assert ev["is_missing"] is True and "target" not in ev and "features" not in ev
    # a missing event that still carries values is a contract violation
    bad = {**ev, "target": 1.0}
    assert not is_valid("raw", bad)


def test_non_missing_requires_values():
    ev = ok_raw(); del ev["features"]
    assert not is_valid("raw", ev)


@pytest.mark.parametrize("mutate", [
    lambda e: e.update(event_time="2010-01-01T00:00:30Z"),    # seconds != 0
    lambda e: e.update(event_time="2010-01-01 00:00:00"),     # not ISO-Z
    lambda e: e.update(schema_version="2.0.0"),
    lambda e: e.update(seq=-1),
    lambda e: e.update(extra_field=1),
    lambda e: e.update(target=-0.1),
    lambda e: e.update(meter_id="UCI_001"),
])
def test_invalid_raw_rejected(mutate):
    ev = copy.deepcopy(ok_raw()); mutate(ev)
    with pytest.raises(jsonschema.ValidationError):
        validate("raw", ev)


def test_encode_is_compact_and_rejects_nan():
    assert b" " not in encode(ok_raw())
    ev = ok_raw(); ev["target"] = float("nan")
    with pytest.raises(ValueError):
        encode(ev)


def test_metric_event_valid():
    validate("metric", metric_event("run-1", "kafka", "kafka.consumer_lag", 12, "telegraf-base"))
    assert not is_valid("metric", {**metric_event("r", "kafka", "Bad Name", 1)})
