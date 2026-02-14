#!/usr/bin/env bash
# Run Anchor API from repo root. Loads .env from repo root.
# Reload only when apps/api or config changes (avoids reloads on tests/ml edits).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT/apps/api"
if [ -x "$ROOT/.venv/bin/uvicorn" ]; then
  exec "$ROOT/.venv/bin/uvicorn" api.main:app --reload --reload-dir "$ROOT/apps/api" --reload-dir "$ROOT/config" --host 0.0.0.0 --port 8000
fi
exec uvicorn api.main:app --reload --reload-dir "$ROOT/apps/api" --reload-dir "$ROOT/config" --host 0.0.0.0 --port 8000
