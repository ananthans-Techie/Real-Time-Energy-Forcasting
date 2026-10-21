# Convenience targets. Everything is also available as a cap-* CLI (see README).
-include .env
export
RAW_PARTITIONS ?= 3

.PHONY: setup test synth synth-prepare prepare up up-kraft down reset topics replay-smoke hourly bench dashboards parser-check

setup:           ; pip install -e ".[dev]"
test:            ; python -m pytest -q
synth:           ; cap-synth --start 2009-09-28 --days 110 --out data/raw/synthetic_uci.txt
synth-prepare: synth ; cap-prepare --input data/raw/synthetic_uci.txt
prepare:         ; cap-prepare --download
up:              ; docker compose --profile zk up -d
up-kraft:        ; docker compose --profile kraft up -d
down:            ; docker compose --profile zk --profile kraft down
reset:           ; docker compose --profile zk --profile kraft down -v
topics:          ; cap-topics --raw-partitions $(RAW_PARTITIONS)
replay-smoke:    ; cap-replay --run-id smoke-$$(date +%H%M%S) --limit 2000 --rate 500
hourly:          ; cap-hourly
bench:           ; cap-benchmark
dashboards:      ; python tools/build_dashboards.py
parser-check:    ; python tools/parser_check.py
