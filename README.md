# Anchor

Voice companion backend: weekly event uploads build an **Independence Graph**; GNN/transformer models score fraud/risk, explain via subgraph attribution, and return watchlists to the edge.

## Architecture

- **Edge**: Device emits structured event packets; batch uploads weekly. No audio capture in this repo.
- **Backend**: Supabase (Postgres + Auth + Realtime), FastAPI, LangGraph orchestration.
- **Graph**: Entity + Event nodes; time-aware edges (Δt, count, recency). Supabase = source of truth; PyG = training + scoring; Neo4j = visualization + investigative queries (see [README_EXTENDED.md](README_EXTENDED.md) §2.4).
- **Models**: HGT baseline; GraphGPS/GPSConv; FraudGT-style edge-attribute attention. Explainability: GNNExplainer + PGExplainer/SubgraphX.

## Repo layout

```
apps/
  api/          FastAPI backend, Supabase, LangGraph pipelines
  worker/       Async jobs: ingest → graph → train/inference
  web/          Next.js 14 dashboard (alerts, sessions, graph, replay, agents)
ml/             PyG models (HGT, GPS, FraudGT), explainers, training
db/             bootstrap_supabase.sql (run once), migrations/ (001–007),
               repair_households_users.sql, drop_signup_trigger.sql
docs/           SUPABASE_SETUP, NEO4J_SETUP, api_ui_contracts, schema, event_packet_spec,
               QUICKSTART_API, SEED_DATA, agents, Modal, GNN audit, demo
scripts/        run_api.sh, run_worker.sh, start_neo4j.sh,
               synthetic_scenarios.py, demo_replay.py, run_financial_agent_demo.py,
               seed_supabase_data.py, run_gnn_e2e.py, run_migration.py, run_replay_time_to_flag.py
config/         settings.py, graph.py, graph_policy.py
tests/          Pytest suite (config, ML, API, pipeline, worker, Modal, routers, GNN e2e)
```

## Modal

Serverless ML training runs on [Modal](https://modal.com). The root **`modal_app.py`** is a minimal entrypoint (`modal run modal_app.py`). Actual training uses the ML apps:

```bash
pip install -e ".[ml]" && modal setup

# HGT training (remote)
modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml --data-dir data/synthetic

# Elliptic training (remote)
modal run ml/modal_train_elliptic.py -- --dataset elliptic --model fraud_gt_style --data-dir data/elliptic --output runs/elliptic
```

Or use **Makefile:** `make modal-train`, `make modal-train-elliptic`. Use `modal.Secret` for Supabase and other env in the cloud. See [docs/modal_training.md](docs/modal_training.md).

---

## Quick start (from repo root)

```bash
# 1. Create venv and install (add [ml] for PyTorch/PyG and GNN)
python3 -m venv .venv && .venv/bin/pip install -e ".[ml]"

# 2. Start the API
./scripts/run_api.sh
# → http://127.0.0.1:8000  (try /health and /docs)

# 3. (Optional) Start the web dashboard — in another terminal:
cd apps/web && npm install && npm run dev
# → http://localhost:3000
```

**API + frontend in two terminals:** See [docs/QUICKSTART_API.md](docs/QUICKSTART_API.md). For UI-only without the API, use **Demo mode** in the app (fixtures).

**Pipeline (one-off, no Supabase required for dry run):**
```bash
./scripts/run_worker.sh --once --household-id hh1
```

## One-command demo (for judges)

Single entrypoint that seeds a synthetic scenario, runs the pipeline, and writes all demo artifacts:

```bash
PYTHONPATH=".:apps/api" python3 scripts/demo_replay.py
```

**Outputs** (in `demo_out/` by default):

- `risk_chart.json` — risk score timeline for charts
- `explanation_subgraph.json` — evidence subgraph (nodes/edges)
- `agent_trace.json` — pipeline step trace
- `scenario_replay.json` — combined payload for the replay UI

**Optional:** update the web UI fixtures and launch the dashboard in demo mode:

```bash
PYTHONPATH=".:apps/api" python3 scripts/demo_replay.py --ui --launch-ui
```

Then open http://localhost:3000 and use **Scenario Replay** (and toggle **Demo mode ON** if needed).

## GNN: train and verify (similar incidents, embeddings, model subgraph)

To have **real** GNN behavior (embeddings, similar incidents, embedding-centroid watchlists, model evidence subgraph), a checkpoint must exist. One-time setup:

```bash
# Train HGT baseline (writes runs/hgt_baseline/best.pt)
python -m ml.train --epochs 8

# Run e2e: pipeline with demo events + assertions
python scripts/run_gnn_e2e.py --skip-train

# Or train if missing then run
python scripts/run_gnn_e2e.py --train
```

Pytest (with checkpoint present):

```bash
pytest tests/test_gnn_e2e.py -v
```

Without a checkpoint, the pipeline still runs but uses fallback scores (no embeddings, no model_subgraph, no embedding-centroid watchlist). See [docs/api_ui_contracts.md](docs/api_ui_contracts.md) for `GET /risk_signals/{id}/similar` and `available: false` when the model did not run.

## How to run (local dev)

### 1. Supabase (local or hosted)

- **Hosted (recommended):** [docs/SUPABASE_SETUP.md](docs/SUPABASE_SETUP.md) — create project, run **`db/bootstrap_supabase.sql`** in SQL Editor once, set `.env`, link first user (Option C: signup → `POST /households/onboard`).
- **If signup returns 500:** Run **`db/drop_signup_trigger.sql`** in SQL Editor (removes Auth trigger); ensure **`db/repair_households_users.sql`** has been run so `households`/`users` exist for the API. See SUPABASE_SETUP.md §6.
- **Existing DB, add embedding columns:** Run **`db/migrations/007_risk_signal_embeddings_extended.sql`** in SQL Editor.
- **Local (optional):** `supabase init && supabase start && supabase db push`; set `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` in `.env`.

### 2. Environment

Create a `.env` file in the project root (or copy from `.env.example` if present). For API + worker with Supabase:

- **SUPABASE_URL**, **SUPABASE_SERVICE_ROLE_KEY** — from [Supabase dashboard](https://supabase.com/dashboard) → Project Settings → API (see [docs/SUPABASE_SETUP.md](docs/SUPABASE_SETUP.md)).

The API and worker run without these for `/health` and pipeline dry runs. Optional: **NEO4J_URI**, **NEO4J_USER**, **NEO4J_PASSWORD** for graph visualization (see §2b).

### 2b. Neo4j (optional)

Neo4j is **optional**; the app works without it. Use it for **Graph view** in the dashboard: sync evidence graph to Neo4j and open Neo4j Browser for Cypher exploration.

- **Quick start (Docker):** `./scripts/start_neo4j.sh` (default password `neo4j123`). Set in **`apps/api/.env`** or root `.env`: `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER=neo4j`, `NEO4J_PASSWORD=neo4j123`, then restart the API.
- **Full guide:** [docs/NEO4J_SETUP.md](docs/NEO4J_SETUP.md) — Docker, Desktop, AuraDB, `GET /graph/neo4j-status`, `POST /graph/sync-neo4j`.
- ML training and scoring use PyG only; Neo4j is for visualization and investigation.

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

### 5. Data: synthetic events or full seed

- **Synthetic scenario (scam + normal):**  
  `python scripts/synthetic_scenarios.py --household-id <uuid> --output db/seed_events.json`  
  Then ingest via `POST /ingest/events` or worker.

- **Full demo seed (thousands of rows):** [docs/SEED_DATA.md](docs/SEED_DATA.md) — `scripts/seed_supabase_data.py` creates households, users, devices, sessions, events, entities, risk_signals, watchlists, etc. Requires Supabase `.env`. Options: `--dry-run`, `--output-json`, `--household-id` / `--user-id` for existing household.

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

### 9. Web dashboard (Next.js)

The caregiver/elder UI runs in `apps/web` (Next.js 14, App Router). From repo root:

```bash
cd apps/web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Set `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` and Supabase auth vars in `apps/web/.env.local` (see [apps/web/README.md](apps/web/README.md)). Use **Demo mode** to run off fixtures without the API.

### 10. Financial Security Agent

- **What it does:** Runs a 7-step playbook (ingest → normalize → detect scam-like motifs → investigation bundle → recommendations → watchlist → consent-gated escalation draft → persist). Read-only: recommends only; never moves money.
- **One-command demo (artifacts + optional UI):** `PYTHONPATH=".:apps/api" python3 scripts/demo_replay.py [--ui] [--launch-ui]`
- **Test locally (no API):** `PYTHONPATH=".:apps/api" python3 scripts/run_financial_agent_demo.py`
- **Unit tests:** `pytest tests/test_financial_agent.py -v`
- **API:** `POST /agents/financial/run` (body: `dry_run`, `time_window_days`); see [docs/api_ui_contracts.md](docs/api_ui_contracts.md) and [docs/agents.md](docs/agents.md).

## API contracts for UI

See [docs/api_ui_contracts.md](docs/api_ui_contracts.md) for:

- Auth (Supabase Auth; household-scoped RLS); onboarding: `POST /households/onboard` after sign-up
- REST: households (me, onboard), sessions, events, risk_signals, feedback, watchlists, device sync, ingest
- Realtime: risk_signals stream
- JSON schemas: risk signal card, risk signal detail (subgraph), weekly summary, watchlist item

## Testing

```bash
# From repo root (recommended: use project venv)
pip install -e ".[ml]"
make test
# or: ruff check apps ml tests && pytest tests apps/api/tests apps/worker/tests ml/tests -v
```

The main test suite lives in **`tests/`** (pytest discovers from `tests`, `apps/api/tests`, `apps/worker/tests`, `ml/tests` per pyproject.toml). See [tests/README.md](tests/README.md) for the full test layout (config, ML, API, pipeline, worker, Modal, routers, spec and strict implementation tests).

## Makefile (quick reference)

| Target | Description |
|--------|-------------|
| `make install` | `pip install -e ".[ml]"` |
| `make test` | ruff + pytest |
| `make lint` | ruff check |
| `make dev-api` | Run API from `apps/api` (use `./scripts/run_api.sh` from root for correct PYTHONPATH) |
| `make dev-worker` | Run worker from `apps/worker` (use `./scripts/run_worker.sh` from root for correct PYTHONPATH) |
| `make synth` | Generate synthetic events to `data/synthetic_events.json` |
| `make train` | HGT train in `ml` with `data/synthetic` |
| `make modal-train` | HGT on Modal (remote) |
| `make modal-train-elliptic` | Elliptic FraudGT-style on Modal |

## Further reading

- **In-depth documentation:** [README_EXTENDED.md](README_EXTENDED.md) — architecture, data flow, ML pipeline, agents, schema, event spec, Modal, and development guide.
- **Setup:** [docs/SUPABASE_SETUP.md](docs/SUPABASE_SETUP.md) — create project, run bootstrap SQL, link user to household.
- **API + UI:** [docs/api_ui_contracts.md](docs/api_ui_contracts.md) — REST, WebSocket, and JSON shapes for the web/mobile UI.
- **Schema:** [docs/schema.md](docs/schema.md) — core and derived tables (including embeddings, calibration), RLS.
- **Event packet:** [docs/event_packet_spec.md](docs/event_packet_spec.md) — edge → backend event format and payload variants.
- **Frontend:** [docs/frontend_notes.md](docs/frontend_notes.md) — data objects and endpoints used by the dashboard.
- **Data & ML:** [docs/DATA_AND_NEXT_STEPS.md](docs/DATA_AND_NEXT_STEPS.md) — real data (HGB, Elliptic), training, and checkpoint download from Modal.
- **Agents:** [docs/agents.md](docs/agents.md) — Financial Security Agent and agent APIs.
- **Modal training:** [docs/modal_training.md](docs/modal_training.md) — remote training and Elliptic.
- **Demo:** [docs/DEMO_MOMENTS.md](docs/DEMO_MOMENTS.md) — hackathon demo angles (temporal, subgraph, similar incidents, HITL, edge).
- **GNN product loop:** [docs/GNN_PRODUCT_LOOP_AUDIT.md](docs/GNN_PRODUCT_LOOP_AUDIT.md) — audit of where GNN is used vs rule-based fallbacks.
- **Quick start (API + frontend):** [docs/QUICKSTART_API.md](docs/QUICKSTART_API.md) — run API and dashboard in two terminals.
- **Seeding demo data:** [docs/SEED_DATA.md](docs/SEED_DATA.md) — seed_supabase_data.py, options, and pipeline follow-up.
- **Neo4j (optional):** [docs/NEO4J_SETUP.md](docs/NEO4J_SETUP.md) — Docker, start_neo4j.sh, API env, Graph view sync.

## Tech stack

- **Backend:** Python 3.11, FastAPI, Pydantic; Supabase (Postgres, Auth, Realtime)
- **Web UI:** Next.js 14 (App Router), TypeScript, TailwindCSS, shadcn/ui, React Flow, Recharts
- **ML:** PyTorch 2.2, PyTorch Geometric 2.6; LangGraph (stateful, durable, HITL)
- **Infra:** Modal (serverless ML jobs); CI: ruff, pytest
