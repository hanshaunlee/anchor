#!/usr/bin/env bash
# Run Anchor API from repo root. Loads .env from repo root.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT/apps/api"
if [ -x "$ROOT/.venv/bin/uvicorn" ]; then
  exec "$ROOT/.venv/bin/uvicorn" api.main:app --reload --host 0.0.0.0 --port 8000
fi
exec uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
