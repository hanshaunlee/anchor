#!/usr/bin/env bash
# Run Anchor worker from repo root. Optional: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT:$ROOT/apps:$ROOT/apps/api"
if [ -x "$ROOT/.venv/bin/python" ]; then
  exec "$ROOT/.venv/bin/python" -m worker.main "$@"
fi
exec python3 -m worker.main "$@"
