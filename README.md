# Anchor

**Anchor** is a backend and dashboard for an edge voice companion that helps protect elders from fraud and social engineering. The edge device sends **structured event packets** (no raw audio in this repo)—transcripts, intents, financial events—which are ingested, turned into a household **Independence Graph**, scored for risk by rules and optional GNN models, explained via motifs and subgraphs, and surfaced as **risk signals**, **watchlists**, and recommendations to caregivers and to the device. The system is **read-only** for money: it recommends and flags; it does not execute financial transactions.

---

## 1. How the product fits together

```
┌─────────────┐     batch      ┌──────────────────────────────────────────────────────────┐
│ Edge device │ ──────────────►│  API (FastAPI)  │  Worker (optional cron)                  │
│ (no audio   │  POST /ingest/ │  • Auth (Supabase JWT)                                    │
│  in repo)   │  events       │  • Households, sessions, risk_signals, watchlists, agents  │
└─────────────┘                │  • WebSocket /ws/risk_signals                             │
                              └─────────────────────────┬──────────────────────────────────┘
                                                        │
                                                        ▼
                              ┌──────────────────────────────────────────────────────────┐
                              │  Pipeline (LangGraph)  — single place risk is computed   │
                              │  ingest → normalize → graph_update → [Financial Agent]    │
                              │  → risk_score (GNN or placeholder) → explain             │
                              │  → consent_gate → watchlist → escalation_draft → persist │
                              └─────────────────────────┬──────────────────────────────────┘
                                                        │
          ┌──────────────────────────────────────────────┼──────────────────────────────────────┐
          ▼                                              ▼                                        ▼
┌─────────────────────┐                    ┌─────────────────────┐                    ┌─────────────────────┐
│ Supabase (Postgres) │                    │ PyG (in-memory)     │                    │ Neo4j (optional)     │
│ Source of truth:    │                    │ Build graph from    │                    │ Evidence subgraph   │
│ sessions, events,   │                    │ tables; run HGT     │                    │ for Cypher / UI     │
│ entities, risk_     │                    │ inference; no       │                    │ exploration only   │
│ signals, watchlists │                    │ persistent graph DB │                    │                     │
└─────────────────────┘                    └─────────────────────┘                    └─────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Next.js dashboard   │  Caregiver: alerts, investigate (timeline, graph, similar incidents,
│ (apps/web)          │  feedback), sessions, watchlists, summaries, agents. Elder: simple
│ + WebSocket         │  view + “Share with caregiver.” Demo mode = fixtures, no API.
└─────────────────────┘
```

- **Risk is computed in one place:** the **LangGraph pipeline** (`api/pipeline.py`). It runs when the **worker** runs (e.g. `./scripts/run_worker.sh --once --household-id <id>`) or when you trigger the **Financial Security Agent** via the API (`POST /agents/financial/run`). The pipeline normalizes events into a graph, runs the agent (and/or its own risk_score node), explains, applies consent, and persists risk_signals and watchlists.
- **Agents** are ordered playbooks inside this flow. The **Financial Security Agent** is both a **pipeline node** (runs after graph_update, before risk_score; in that context it does not write to the DB) and an **on-demand API** (POST /agents/financial/run with optional dry_run) that runs the same playbook with Supabase and broadcast. So: “agents” are in the scheme of the product as the **risk + recommendation logic** that sits between “graph built” and “scores + watchlists persisted.”

---

## 2. What technology is used for what

| Technology | Used for |
|------------|----------|
| **Supabase** | Postgres (sessions, events, entities, risk_signals, watchlists, agent_runs, embeddings, calibration), Auth (JWT), RLS (household-scoped). Single source of truth. |
| **FastAPI** | REST API and WebSocket. Routers: households (me, onboard), sessions, risk_signals (list, detail, feedback, similar), watchlists, device/sync, ingest, summaries, graph (evidence, Neo4j sync/status), agents (financial run, status, trace). |
| **LangGraph** | Orchestrates the pipeline: ingest → normalize → graph_update → financial_security_agent → risk_score → explain → consent_gate → (optional needs_review) → watchlist → escalation_draft → persist. State is in-memory (MemorySaver); can be swapped for DB checkpointer. |
| **PyTorch Geometric** | **Graph building** from events/tables (`ml/graph/builder.py` → utterances, entities, mentions, relationships; `build_hetero_from_tables` → HeteroData). **Models:** HGT (main pipeline/agent scoring + embeddings), GraphGPS (available, not default), FraudGT-style (Elliptic pipeline only). **Explainers:** motifs (rules), PGExplainer/GNNExplainer (model subgraph when GNN runs). Training: `ml/train.py` (HGT), `ml/train_elliptic.py` (FraudGT). |
| **Neo4j** | Optional. Evidence subgraph (entities + relationships) is mirrored from API/worker for **visualization and Cypher** in the dashboard (Graph view → Sync to Neo4j → Open in Neo4j Browser). Not used for training or scoring. |
| **Next.js (apps/web)** | Dashboard: auth (Supabase), onboarding, alerts (list + investigate: timeline, graph, similar incidents, feedback), sessions, watchlists, summaries, graph view, ingest, agents (dry run, trace), elder view, scenario replay. React Flow for graph, Recharts for charts. WebSocket for live risk signals. |
| **Modal** | Serverless runs for **training** (HGT, Elliptic); not for the main API or pipeline. `make modal-train`, `make modal-train-elliptic`. |
| **config/** | Pipeline thresholds and consent keys (`settings.py`), graph schema and motif keywords (`graph.py`), ASR/intent confidence gate for graph mutation (`graph_policy.py`). |

---

## 3. Agents in the product

There is one agent: **Financial Security** (`financial_security`).

- **What it does:** Ingest events (from DB or pre-filled) → normalize with GraphBuilder → detect risk (rule/motif score + optional GNN score when checkpoint exists) → build investigation bundle (motif_tags, timeline_snippet, subgraph) → recommendations (checklist) → watchlist synthesis → consent-gated escalation draft → persist risk_signals (and optionally broadcast) + watchlists + agent_runs row with step_trace.
- **Where it runs:**  
  - **Inside the pipeline** (node `financial_security_agent`): uses pipeline state only; no Supabase in context, so no DB write. Good for “run full pipeline and only persist at the end.”  
  - **On-demand via API** (`POST /agents/financial/run`): runs the same playbook with Supabase; can persist and broadcast. Optional `dry_run` (preview only) and `use_demo_events` (synthetic scam scenario).  
- **API:** `POST /agents/financial/run`, `GET /agents/financial/demo` (no auth), `GET /agents/status`, `GET /agents/financial/trace?run_id=`.  
- Details: [docs/agents.md](docs/agents.md).

---

## 4. Models and GNN

- **HGT** (`ml/models/hgt_baseline.py`): Main model. Heterogeneous graph over sessions, utterances, entities, events; used for **entity-level risk scores** and **embeddings** (pooled hidden states). When a checkpoint exists, the pipeline and the Financial Agent use it; embeddings power **similar incidents** and **embedding-centroid watchlists**. Trained with `ml/train.py` (synthetic or HGB); inference in `ml/inference.py` and inside the pipeline.
- **GraphGPS** (`ml/models/gps_model.py`): Available for training; not the default pipeline model.
- **FraudGT-style** (`ml/models/fraud_gt_style.py`): Used in the **Elliptic** pipeline only (`ml/train_elliptic.py`); not used for the voice/entity pipeline.
- **When the GNN is used:** Pipeline loads HGT from `config.settings.checkpoint_path`. If present: real inference + PGExplainer model_subgraph for high-scoring nodes; embeddings stored for similar incidents. If missing/fails: placeholder scores, no embeddings, no model_subgraph; similar incidents return `available: false, reason: "model_not_run"`. Financial Agent combines rule + model when GNN ran (`0.6*rule + 0.4*model`); otherwise rule only. See [docs/GNN_PRODUCT_LOOP_AUDIT.md](docs/GNN_PRODUCT_LOOP_AUDIT.md).

---

## 5. Frontend: what users can do

- **Auth:** Sign up, sign in, onboard (create household after sign-up), sign out.
- **Caregiver/Admin:** Dashboard (feed, risk chart, latest signals); **Alerts** (list, filter, real-time via WebSocket); **Investigate** one alert (timeline, graph evidence, motif tags, recommended actions, similar incidents, feedback: Confirm scam / False alarm / Unsure, agent trace); **Sessions** (list by date, open session → events, consent); **Watchlists**; **Summaries** (weekly, trends); **Graph** (evidence subgraph, Sync to Neo4j, Open in Neo4j Browser); **Ingest** (batch upload); **Agents** (dry run Financial Agent, view trace).
- **Elder:** Elder view (simple summary, recommendation, “Share with caregiver” toggle; no raw alerts/graph).
- **Scenario Replay:** Animate scam storyline (score chart, graph, trace) from demo fixtures.
- **Demo mode:** Sidebar toggle or `NEXT_PUBLIC_DEMO_MODE=true`; app uses local fixtures, no API.

Setup: [apps/web/README.md](apps/web/README.md). Contracts: [docs/api_ui_contracts.md](docs/api_ui_contracts.md), [docs/frontend_notes.md](docs/frontend_notes.md).

---

## 6. Repo layout

```
apps/api/       FastAPI app, routers, pipeline (LangGraph), graph_state, broadcast, deps, schemas; domain (ingest, risk, explain, similarity, watchlist, graph, agents)
apps/worker/    Entrypoint + jobs: ingest from Supabase, run pipeline, persist risk_signals/embeddings/watchlists/agent_runs, optional Neo4j sync
apps/web/       Next.js 14 dashboard (see above)
ml/             Models (HGT, GPS, FraudGT), graph builder + subgraph + time_encoding, explainers (motifs, gnn, pg), train/inference, continual, cache, Modal training
config/         settings (pipeline + ML), graph (schema, keywords), graph_policy (confidence gate)
db/             bootstrap_supabase.sql (run once), migrations 001–007, repair_households_users.sql, drop_signup_trigger.sql
scripts/        run_api.sh, run_worker.sh, start_neo4j.sh, synthetic_scenarios.py, demo_replay.py, seed_supabase_data.py, run_gnn_e2e.py, run_financial_agent_demo.py, run_migration.py, run_replay_time_to_flag.py
tests/          Pytest: config, ML, API, pipeline, worker, Modal, agents, GNN e2e, spec, strict
docs/           SUPABASE_SETUP, NEO4J_SETUP, api_ui_contracts, schema, event_packet_spec, agents, modal_training, DATA_AND_NEXT_STEPS, frontend_notes, GNN_PRODUCT_LOOP_AUDIT, QUICKSTART_API, SEED_DATA, DEMO_MOMENTS
```

---

## 7. Quick start and running things

```bash
# Install (add [ml] for GNN)
python3 -m venv .venv && .venv/bin/pip install -e ".[ml]"

# API (from repo root)
./scripts/run_api.sh
# → http://127.0.0.1:8000  (/health, /docs)

# Dashboard (other terminal)
cd apps/web && npm install && npm run dev
# → http://localhost:3000
```

- **Pipeline once:** `./scripts/run_worker.sh --once --household-id <uuid>`
- **Demo (no API):** Toggle Demo mode in the app, or run `PYTHONPATH=".:apps/api" python3 scripts/demo_replay.py --ui --launch-ui` and open Scenario Replay.
- **Supabase:** [docs/SUPABASE_SETUP.md](docs/SUPABASE_SETUP.md) — create project, run `db/bootstrap_supabase.sql` once, set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in `.env` (root or `apps/api`). Onboard after sign-up via app or `POST /households/onboard`.
- **Neo4j (optional):** `./scripts/start_neo4j.sh`; set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` in `.env`. [docs/NEO4J_SETUP.md](docs/NEO4J_SETUP.md).
- **Train HGT:** `make train` or `python -m ml.train --config ml/configs/hgt_baseline.yaml --data-dir data/synthetic`. Remote: `make modal-train`. Elliptic: `make modal-train-elliptic`.
- **Test:** `make test`. [tests/README.md](tests/README.md).
- **Makefile:** install, test, lint, dev-api, dev-worker, synth, train, modal-train, modal-train-elliptic.

---

## 8. Docs and codebase reference

| Need | Where |
|------|--------|
| Full file-by-file reference | [README_EXTENDED.md](README_EXTENDED.md) |
| Supabase + Neo4j setup | [docs/SUPABASE_SETUP.md](docs/SUPABASE_SETUP.md), [docs/NEO4J_SETUP.md](docs/NEO4J_SETUP.md) |
| API and UI contracts | [docs/api_ui_contracts.md](docs/api_ui_contracts.md), [docs/frontend_notes.md](docs/frontend_notes.md) |
| Schema, event packet | [docs/schema.md](docs/schema.md), [docs/event_packet_spec.md](docs/event_packet_spec.md) |
| Agents, Modal, data/next steps | [docs/agents.md](docs/agents.md), [docs/modal_training.md](docs/modal_training.md), [docs/DATA_AND_NEXT_STEPS.md](docs/DATA_AND_NEXT_STEPS.md) |
| GNN vs rule fallbacks | [docs/GNN_PRODUCT_LOOP_AUDIT.md](docs/GNN_PRODUCT_LOOP_AUDIT.md) |
| Quick start API + UI | [docs/QUICKSTART_API.md](docs/QUICKSTART_API.md) |
| Seed data, demo | [docs/SEED_DATA.md](docs/SEED_DATA.md), [docs/DEMO_MOMENTS.md](docs/DEMO_MOMENTS.md) |

---

**Tech stack (summary):** Python 3.11, FastAPI, Pydantic, Supabase (Postgres + Auth), LangGraph, PyTorch 2.2, PyTorch Geometric 2.6 (HGT, GPS, FraudGT-style), Next.js 14 (TypeScript, Tailwind, shadcn/ui, React Flow, Recharts), Modal (training), ruff, pytest.
