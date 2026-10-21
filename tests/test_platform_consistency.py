"""The frozen schema, the Telegraf mapping, the compose file and the dashboards must agree."""
import json
import re
import tomllib
from pathlib import Path

import yaml

from capstone import constants as C

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "src" / "capstone" / "schemas"


def telegraf():
    return tomllib.loads((ROOT / "telegraf" / "telegraf.conf").read_text())


def test_telegraf_raw_mapping_covers_schema():
    raw_in = telegraf()["inputs"]["kafka_consumer"][0]
    assert raw_in["topics"] == [C.TOPIC_RAW]
    j = raw_in["json_v2"][0]
    assert j["measurement_name"] == C.MEASUREMENT_RAW and j["timestamp_path"] == "event_time"
    schema = json.loads((SCHEMAS / "energy_raw.v1.schema.json").read_text())
    props = set(schema["properties"])
    mapped = {f["path"] for f in j["field"]} | {t["path"] for t in j["tag"]} | {j["timestamp_path"]}
    flat = set()
    for p in mapped:
        flat.add(p.split(".")[0] if p.startswith("features.") else p)
    # every schema property is either mapped or deliberately constant/derivable
    assert props - flat <= {"schema_version", "event_id"}, props - flat
    # feature fields: names and rename targets
    feats = {f["path"].split(".")[1]: f.get("rename") for f in j["field"] if f["path"].startswith("features.")}
    assert set(feats) == set(C.FEATURES) and all(k == v for k, v in feats.items())
    # missing-minute handling: every measurement field is optional except the always-present ones
    always = {"is_missing", "seq", "replay_time_ms"}
    for f in j["field"]:
        assert f.get("optional", False) or f["path"] in always, f["path"]
    assert {t["path"] for t in j["tag"]} == {"meter_id", "run_id"}


def test_telegraf_metric_mapping_matches_schema():
    m_in = telegraf()["inputs"]["kafka_consumer"][1]
    assert m_in["topics"] == [C.TOPIC_METRICS] and m_in["consumer_group"] == C.GROUP_METRICS
    j = m_in["json_v2"][0]
    assert j["measurement_name"] == C.MEASUREMENT_METRIC and j["timestamp_path"] == "metric_time_ms"
    schema = json.loads((SCHEMAS / "pipeline_metric.v1.schema.json").read_text())
    tags = {t["path"] for t in j["tag"]}
    assert tags | {"value", "metric_time_ms", "schema_version"} == set(schema["properties"])


def test_telegraf_env_vars_are_supplied_by_compose():
    conf = (ROOT / "telegraf" / "telegraf.conf").read_text()
    used = set(re.findall(r"\$\{([A-Z_]+)\}", conf))
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    supplied = set(compose["services"]["telegraf"]["environment"])
    assert used <= supplied, used - supplied
    assert telegraf()["inputs"]["kafka_consumer"][0]["consumer_group"] == "${TELEGRAF_CONSUMER_GROUP}"
    assert compose["services"]["telegraf"]["environment"]["TELEGRAF_CONSUMER_GROUP"].endswith(C.GROUP_BASE + "}")


def test_compose_pins_influx_v2_and_both_kafka_variants_share_hostname():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    assert compose["influxdb"]["image"].startswith("influxdb:2")          # never :latest (v3)
    for k in ("kafka-zk", "kafka-kraft"):
        assert "kafka" in compose[k]["networks"]["default"]["aliases"]
        assert compose[k]["environment"]["KAFKA_AUTO_CREATE_TOPICS_ENABLE"] == "false"
    assert "zk" in compose["zookeeper"]["profiles"] and "kraft" in compose["kafka-kraft"]["profiles"]


def test_dashboards_only_query_contract_fields():
    allowed_raw = set(C.VALUE_COLUMNS) | {"is_missing", "seq", "replay_time_ms"}
    d = json.loads((ROOT / "grafana" / "dashboards" / "energy_readings.json").read_text())
    for p in d["panels"]:
        for t in p.get("targets", []):
            used = set(re.findall(r'r\._field == "([a-z_0-9]+)"', t["query"]))
            assert used <= allowed_raw, (p["title"], used - allowed_raw)
    h = json.loads((ROOT / "grafana" / "dashboards" / "pipeline_health.json").read_text())
    names = set()
    for p in h["panels"]:
        for t in p.get("targets", []):
            names |= set(re.findall(r'r\.name == "([a-z_.0-9]+)"', t["query"]))
    assert {"producer.rate_rps", "kafka.consumer_lag", "e2e.latency_ms",
            "docker.cpu_percent", "docker.mem_mib"} <= names
