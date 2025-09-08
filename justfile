default:
    @just --list

set dotenv-load := true

setup:
    python -m venv .venv
    source .venv/bin/activate && pip install -r requirements.txt

bench:
    source .venv/bin/activate && python tools/bench.py

fmt:
    ruff check . --fix || true
    ruff format .

lab_2:
    source .venv/bin/activate && python labs/lab_2.py
