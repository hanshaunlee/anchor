# Anchor

Voice companion backend: weekly event uploads build an **Independence Graph**; GNN/transformer models score fraud/risk, explain via subgraph attribution, and return watchlists to the edge.

## Architecture

- **Edge**: Device emits structured event packets; batch uploads weekly. No audio capture in this repo.
- **Backend**: Supabase (Postgres + Auth + Realtime), FastAPI, LangGraph orchestration.
- **Graph**: Entity + Event nodes; time-aware edges (Δt, count, recency). Stored in PyG for ML; Neo4j optional for UI.
- **Models**: HGT baseline; GraphGPS/GPSConv; FraudGT-style edge-attribute attention. Explainability: GNNExplainer + PGExplainer/SubgraphX.

## Repo layout

```
apps/
  api/          FastAPI backend, Supabase, LangGraph pipelines
  worker/       Async jobs: ingest → graph → train/inference
ml/             PyG models (HGT, GPS, FraudGT), explainers, training
db/             Supabase SQL migrations + RLS
docs/           Schema, API, event packet spec, model docs
scripts/        Dataset prep, synthetic scenario generator
```

## Modal

Serverless jobs (workers, ML) run on [Modal](https://modal.com). After `pip install -e .` and `modal setup`:

```bash
modal run modal_app.py      # run once
modal deploy modal_app.py   # deploy to workspace
modal serve modal_app.py    # dev with hot reload
```

Use `modal.Secret` for Supabase and other env in the cloud.

---

## Quick start (from repo root)

```bash
# 1. Create venv and install (optional: add [ml] for PyTorch/PyG)
python3 -m venv .venv && .venv/bin/pip install -e .

# 2. Start the API (from repo root)
./scripts/run_api.sh
# → http://127.0.0.1:8000  (try /health and /docs)

# 3. Run the pipeline once (no Supabase needed for a dry run)
./scripts/run_worker.sh --once --household-id hh1
```

## How to run (local dev)

### 1. Supabase local (optional)

```bash
cd /path/to/Anchor
supabase init   # if not already
supabase start
supabase db push
# Then set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY in .env
```

### 2. Environment

```bash
cp .env.example .env
# Optional: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY (API/worker work without for /health and pipeline dry run)
```

### 3. API

```bash
# From repo root (uses .venv if present)
./scripts/run_api.sh
# Or manually:
# PYTHONPATH=apps/api uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Worker

```bash
./scripts/run_worker.sh
# Run pipeline once: ./scripts/run_worker.sh --once --household-id <uuid>
```

### 5. Synthetic data + pipeline

```bash
# Generate synthetic events (normal + scam scenario)
python scripts/synthetic_scenarios.py --household-id <uuid> --output db/seed_events.json

# Ingest and run pipeline (via API or worker)
# POST /ingest/events with batch, or worker cron for ingest_events_batch
```

### 6. Train baseline model

```bash
cd ml && python train.py --config configs/hgt_baseline.yaml --data-dir data/synthetic
```

### 7. Run inference

```bash
cd ml && python inference.py --checkpoint runs/hgt_baseline/best.pt --household-id <uuid>
```

### 8. View risk_signals

- API: `GET /risk_signals?household_id=...`
- Realtime: connect to `WS /ws/risk_signals`

### 9. Financial Security Agent

- **What it does:** Runs a 7-step playbook (ingest → normalize → detect scam-like motifs → investigation bundle → recommendations → watchlist → consent-gated escalation draft → persist). Read-only: recommends only; never moves money.
- **Test locally (no API):** `PYTHONPATH=".:apps/api" python scripts/run_financial_agent_demo.py`
- **Unit tests:** `pytest tests/test_financial_agent.py -v`
- **API:** `POST /agents/financial/run` (body: `dry_run`, `time_window_days`); see `docs/agents.md`.

## API contracts for UI

See `docs/api_ui_contracts.md` for:

- Auth (Supabase Auth; household-scoped RLS)
- REST: households, sessions, events, risk_signals, feedback, watchlists, device sync, ingest
- Realtime: risk_signals stream
- JSON schemas: risk signal card, risk signal detail (subgraph), weekly summary, watchlist item

## Testing

```bash
# From repo root (recommended: use project venv)
pip install -e ".[ml]"
make test
# or: ruff check apps ml tests && pytest tests -v
```

See `tests/README.md` for the full test layout (config, ML, API, pipeline, worker, Modal, routers, spec tests).

## Makefile (quick reference)

| Target | Description |
|--------|-------------|
| `make install` | `pip install -e ".[ml]"` |
| `make test` | ruff + pytest |
| `make lint` | ruff check |
| `make dev-api` | Run API from `apps/api` |
| `make dev-worker` | Run worker |
| `make synth` | Generate synthetic events to `data/synthetic_events.json` |
| `make train` | HGT train in `ml` with `data/synthetic` |
| `make modal-train` | HGT on Modal (remote) |
| `make modal-train-elliptic` | Elliptic FraudGT-style on Modal |

## Further reading

- **In-depth documentation:** [README_EXTENDED.md](README_EXTENDED.md) — architecture, data flow, ML pipeline, agents, schema, event spec, Modal, and development guide.
- **Docs:** `docs/` — API contracts, agents, schema, event packet spec, Modal training, data & next steps.

## Tech stack

- Python 3.11, FastAPI, Pydantic
- Supabase (Postgres, Auth, Realtime)
- PyTorch 2.2, PyTorch Geometric 2.6
- LangGraph (stateful, durable, HITL)
- Modal (serverless ML jobs)
- CI: ruff, pytest
