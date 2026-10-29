"""Unit and integration tests for Workstream 3 Anomaly Detection.

Validates:
  - Non-parametric Rolling Median and MAD calculations.
  - Positive spike detection without corrupting baseline history.
  - Zero-dropout and missing data detection.
  - Multi-meter independence.
  - Full schema compliance for energy.anomalies events.
"""
import jsonschema
import numpy as np
import pytest

from cap_anomaly import AnomalyDetector, RollingMADDetector
from capstone import events as E


def test_rolling_mad_baseline_learning():
    det = RollingMADDetector(window_size=24, threshold=3.5, min_samples=5)
    # Feed 10 steady readings around 1.0 kW
    for i in range(10):
        val = 1.0 + 0.05 * np.sin(i)
        exp, score, is_anom, anom_type = det.update(val)
        if i < 4:
            assert anom_type == "normal"
            assert is_anom is False
        else:
            assert is_anom is False
            assert anom_type == "normal"
            assert abs(exp - 1.0) < 0.1
            assert score < 3.5


def test_point_spike_detection():
    det = RollingMADDetector(window_size=24, threshold=3.5, min_samples=5)
    for _ in range(15):
        det.update(1.0)

    # Inject extreme spike
    exp, score, is_anom, anom_type = det.update(5.5)
    assert is_anom is True
    assert anom_type == "spike"
    assert score >= 3.5
    assert exp == 1.0

    # Verify spike did NOT contaminate rolling baseline
    exp2, score2, is_anom2, anom_type2 = det.update(1.02)
    assert is_anom2 is False
    assert anom_type2 == "normal"
    assert exp2 == 1.0


def test_zero_dropout_detection():
    det = RollingMADDetector(window_size=24, threshold=3.5, min_samples=5)
    for _ in range(10):
        det.update(1.5)

    # Inverter trip / zero dropout
    exp, score, is_anom, anom_type = det.update(0.0)
    assert is_anom is True
    assert anom_type == "dropout"
    assert exp == 1.5


def test_missing_reading_detection():
    det = RollingMADDetector(window_size=24, threshold=3.5, min_samples=5)
    for _ in range(8):
        det.update(1.2)

    # Missing reading
    exp, score, is_anom, anom_type = det.update(None)
    assert is_anom is True
    assert anom_type == "missing"


def test_anomaly_detector_process_record_and_schema():
    detector = AnomalyDetector(window_size=24, threshold=3.5)

    # Seed baseline
    for h in range(10):
        rec = {
            "run_id": "C_adaptive-r1",
            "event_id": f"uci-001-H20100101T{h:02d}0000Z",
            "meter_id": "uci-001",
            "event_time": f"2010-01-01T{h:02d}:00:00Z",
            "target": 1.1 + 0.02 * (h % 3),
            "is_missing": False,
            "trigger_replay_time_ms": 1792522900000 + h * 1000,
        }
        ev = detector.process_record(rec)
        assert ev is not None
        assert ev["is_anomaly"] is False
        E.validate("anomaly", ev)

    # Spike record
    spike_rec = {
        "run_id": "C_adaptive-r1",
        "event_id": "uci-001-H20100101T100000Z",
        "meter_id": "uci-001",
        "event_time": "2010-01-01T10:00:00Z",
        "target": 5.8,
        "is_missing": False,
        "trigger_replay_time_ms": 1792522910000,
    }
    ev_spike = detector.process_record(spike_rec)
    assert ev_spike is not None
    assert ev_spike["is_anomaly"] is True
    assert ev_spike["anomaly_type"] == "spike"
    assert ev_spike["event_id"] == "uci-001-A20100101T100000Z"
    assert ev_spike["trigger_event_id"] == "uci-001-H20100101T100000Z"
    E.validate("anomaly", ev_spike)


def test_anomaly_schema_rejects_invalid_event():
    ev = E.build_anomaly_event(
        run_id="C_adaptive-r1",
        meter_id="uci-001",
        event_time_iso="2010-01-01T12:00:00Z",
        target_value=4.5,
        expected_value=1.1,
        anomaly_score=5.2,
        is_anomaly=True,
        trigger_event_id="uci-001-H20100101T120000Z",
        trigger_replay_time_ms=1792500000000,
    )
    # Should validate
    E.validate("anomaly", ev)

    # Reject unexpected extra property
    bad_ev = dict(ev)
    bad_ev["unexpected_property"] = "invalid"
    with pytest.raises(jsonschema.ValidationError):
        E.validate("anomaly", bad_ev)


def test_multi_meter_anomaly_isolation():
    detector = AnomalyDetector(window_size=24, threshold=3.5)

    # meter 1 gets baseline ~ 1.0 kW
    for h in range(10):
        detector.process_record({
            "run_id": "C_adaptive-r1",
            "event_id": f"uci-001-H20100101T{h:02d}0000Z",
            "meter_id": "uci-001",
            "event_time": f"2010-01-01T{h:02d}:00:00Z",
            "target": 1.0,
            "trigger_replay_time_ms": 1792500000000,
        })

    # meter 2 gets baseline ~ 4.0 kW
    for h in range(10):
        detector.process_record({
            "run_id": "C_adaptive-r1",
            "event_id": f"uci-002-H20100101T{h:02d}0000Z",
            "meter_id": "uci-002",
            "event_time": f"2010-01-01T{h:02d}:00:00Z",
            "target": 4.0,
            "trigger_replay_time_ms": 1792500000000,
        })

    # 4.0 kW is an extreme anomaly for meter 1
    ev1 = detector.process_record({
        "run_id": "C_adaptive-r1",
        "event_id": "uci-001-H20100101T110000Z",
        "meter_id": "uci-001",
        "event_time": "2010-01-01T11:00:00Z",
        "target": 4.0,
        "trigger_replay_time_ms": 1792500000000,
    })
    assert ev1["is_anomaly"] is True

    # But 4.0 kW is completely normal for meter 2
    ev2 = detector.process_record({
        "run_id": "C_adaptive-r1",
        "event_id": "uci-002-H20100101T110000Z",
        "meter_id": "uci-002",
        "event_time": "2010-01-01T11:00:00Z",
        "target": 4.05,
        "trigger_replay_time_ms": 1792500000000,
    })
    assert ev2["is_anomaly"] is False
