"""Offline check of the Telegraf parsing/transform/reconstruct mapping -- no Kafka, no InfluxDB.

  python tools/parser_check.py            # writes runs/parser_check/{telegraf.conf,sample.jsonl}
  docker run --rm -v "$PWD/runs/parser_check:/pc:ro" telegraf:1.32 telegraf --config /pc/telegraf.conf
  (let it print for ~5 s, then Ctrl-C)

It reuses the *exact* json_v2 block from telegraf/telegraf.conf, reads the sample events with the
tail input, and prints InfluxDB line protocol.  Expected:
  * ordinary minute  -> energy_raw,meter_id=..,run_id=.. target=..,voltage=..,...,is_missing=false,seq=..i,replay_time_ms=..i <event_time ns>
  * missing minute   -> energy_raw,... is_missing=true,seq=..i,replay_time_ms=..i <event_time ns>   (no target/feature fields)
If a missing minute is dropped or errors, set optional=true on the failing field or tell the data owner.
"""
import json
import re
import sys
from pathlib import Path

from capstone.replay_producer import load_config, run_replay

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "parser_check"


def build_conf() -> str:
    conf = (ROOT / "telegraf" / "telegraf.conf").read_text()
    start = conf.index('  data_format = "json_v2"')
    end = conf.index("# ------------------------------------------- pipeline metrics")
    block = conf[start:end].replace("inputs.kafka_consumer", "inputs.tail")
    return ('[agent]\n  flush_interval = "1s"\n  omit_hostname = true\n\n'
            '[[inputs.tail]]\n  files = ["/pc/sample.jsonl"]\n  from_beginning = true\n  watch_method = "poll"\n'
            + block + '\n[[outputs.file]]\n  files = ["stdout"]\n  data_format = "influx"\n')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src = ROOT / "data" / "processed" / "uci_minute.parquet"
    if not src.exists():
        print("Run cap-prepare first (data/processed/uci_minute.parquet missing).", file=sys.stderr); return 2
    import pandas as pd
    df = pd.read_parquet(src)
    miss_idx = list(df.index[df["is_missing"]][:3])
    ok_idx = list(df.index[~df["is_missing"]][:3])
    mini = df.loc[sorted(set(ok_idx + miss_idx))].reset_index(drop=True)
    mini_path = OUT / "mini_minute.parquet"
    mini.to_parquet(mini_path)
    cfg = load_config(None, {"run": {"run_id": "pc"}, "source": {"path": str(mini_path), "split": "all"},
                             "profile": {"mode": "max"}, "sink": "jsonl", "output_dir": str(OUT / "tmp")})
    run_replay(cfg, quiet=True)
    lines = (OUT / "tmp" / "pc" / "events.jsonl").read_text().splitlines()
    (OUT / "sample.jsonl").write_text("\n".join(lines) + "\n")
    miss = [l for l in lines if json.loads(l)["is_missing"]]
    (OUT / "telegraf.conf").write_text(build_conf())
    print(f"wrote {OUT}/sample.jsonl ({len(lines)} events, {len(miss)} missing) and telegraf.conf\n"
          f'docker run --rm -v "{OUT}:/pc:ro" telegraf:1.32 telegraf --config /pc/telegraf.conf')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
