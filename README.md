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

- **Risk is computed in one place:** the **shared risk scoring service** (`domain/risk_scoring_service.py`) used by the **LangGraph pipeline**, the **worker**, and the **Financial Security Agent**. No silent placeholders: when the GNN is unavailable, the API returns `model_available=false` and explicit rule-only fallback. The pipeline runs when the **worker** runs (e.g. `./scripts/run_worker.sh --once --household-id <id>`) or when you trigger an agent via the API. **Event ingestion** is idempotent on `(session_id, seq)`; **normalize** is deterministic (events sorted by session and seq).
- **Agents** are ordered playbooks. The **Financial Security Agent** is both a pipeline node and an on-demand API (`POST /agents/financial/run`). Additional agents: **Graph Drift**, **Evidence Narrative**, **Ring Discovery**, **Continual Calibration**, **Synthetic Red-Team** — each has status, last run, dry-run preview, and trace under `/agents/*`.

---

## 2. What technology is used for what

| Technology | Used for |
|------------|----------|
| **Supabase** | Postgres (sessions, events with UNIQUE(session_id, seq), entities, risk_signals, watchlists, agent_runs with step_trace and summary_json, risk_signal_embeddings with optional pgvector, **rings** and **ring_members** (migration 009), **household_calibration** with calibration_params/last_calibrated_at (migration 010)), Auth (JWT), RLS. Optional pgvector for similar-incidents (migration 008). Single source of truth. |
| **FastAPI** | REST API and WebSocket. Routers: households, sessions, risk_signals (list, detail, feedback, similar with retrieval_provenance), watchlists, device/sync, ingest (idempotent upsert on session_id, seq), summaries, graph, **agents** (financial run/demo/trace; drift, narrative, ring, calibration, redteam run; status; generic trace). |
| **LangGraph** | Pipeline: ingest → normalize (deterministic) → graph_update → financial_security_agent → risk_score (shared scoring service) → explain (model_evidence_quality, stable entity IDs) → consent_gate → watchlist (embedding-centroid when GNN ran) → escalation_draft → persist. State in-memory (MemorySaver). |
| **PyTorch Geometric** | **Graph building** from events/tables (`ml/graph/builder.py` → utterances, entities, mentions, relationships; `build_hetero_from_tables` → HeteroData). **Models:** HGT (main pipeline/agent scoring + embeddings), GraphGPS (available, not default), FraudGT-style (Elliptic pipeline only). **Explainers:** motifs (rules), PGExplainer/GNNExplainer (model subgraph when GNN runs). Training: `ml/train.py` (HGT), `ml/train_elliptic.py` (FraudGT). |
| **Neo4j** | Optional. Evidence subgraph (entities + relationships) is mirrored from API/worker for **visualization and Cypher** in the dashboard (Graph view → Sync to Neo4j → Open in Neo4j Browser). Not used for training or scoring. |
| **Next.js (apps/web)** | Dashboard: auth (Supabase), onboarding, alerts (list + investigate: timeline, graph, similar incidents, feedback), sessions, watchlists, summaries, graph view, ingest, agents (dry run, trace), elder view, scenario replay. React Flow for graph, Recharts for charts. WebSocket for live risk signals. |
| **Modal** | Serverless runs for **training** (HGT, Elliptic); not for the main API or pipeline. `make modal-train`, `make modal-train-elliptic`. |
| **config/** | Pipeline thresholds and consent keys (`settings.py`), graph schema and motif keywords (`graph.py`), ASR/intent confidence gate for graph mutation (`graph_policy.py`). |

---

## 3. Agents in the product

**Financial Security** (`financial_security`) — main playbook: ingest → normalize → detect risk (shared scoring service + motifs) → investigation bundle → watchlist synthesis → persist. Runs as pipeline node or on-demand: `POST /agents/financial/run`, `GET /agents/financial/demo`, `GET /agents/financial/trace?run_id=`.

**Additional agents** (each with dry-run preview; step_trace and summary_json persisted to `agent_runs`; all use shared `domain/agents/base` and `domain/ml_artifacts`):

| Agent | Purpose | API |
|-------|---------|-----|
| **Graph Drift** | Multi-metric embedding drift (centroid, MMD, PCA+KS, neighbor stability); root-cause (model_change / new_pattern / behavior_shift); open `drift_warning` risk_signal + optional summary when drift &gt; threshold | `POST /agents/drift/run` |
| **Evidence Narrative** | Evidence-grounded narrative (deterministic template + optional LLM); redaction by consent; store in `risk_signals.explanation.summary` and `narrative`; `narrative_evidence_only` for UI badge | `POST /agents/narrative/run` |
| **Ring Discovery** | Interaction graph from relationships + mentions; NetworkX clustering (or Neo4j GDS when enabled); `ring_candidate` risk_signals; persist `rings` and `ring_members` (migration 009) | `POST /agents/ring/run` |
| **Continual Calibration** | Platt scaling / conformal threshold from feedback; update `household_calibration` (calibration_params, last_calibrated_at — migration 010); before/after ECE report | `POST /agents/calibration/run` |
| **Synthetic Red-Team** | Scenario DSL (themes + variants); sandbox pipeline run; regression assertions (similar incidents, evidence subgraph); pass rate and failing_cases in summary | `POST /agents/redteam/run` |

- **Status and trace:** `GET /agents/status` returns last run per agent (all six) with `last_run_summary` (e.g. drift_detected, rings_found, regression_pass_rate). `GET /agents/trace?run_id=&agent_name=` or `GET /agents/{slug}/trace?run_id=` returns step_trace and summary. The dashboard shows **Agent Trace** and, per agent, last-run summary (drift metrics, ring count, red-team pass rate).
- **Scheduled runs:** `scripts/run_agent_cron.py --household-id <uuid> --agent drift|narrative|ring|calibration|redteam` (optional `--no-dry-run`); set `ANCHOR_AGENT_CRON_DRY_RUN=false` to persist.
- Details: [docs/agents.md](docs/agents.md).

---

## 4. ML models: what each helps with

| Model | What it helps with | What it does *not* do |
|-------|--------------------|------------------------|
| **HGT** (`ml/models/hgt_baseline.py`) | **Entity-level risk scores** via the **shared risk scoring service** (pipeline, worker, Financial Agent). **Embeddings** for **similar incidents** (pgvector RPC or JSONB cosine) and **embedding-centroid watchlists** (hard dependency: no centroid when model didn’t run). **Model evidence subgraph** via PGExplainer with **stable DB entity IDs** and **model_evidence_quality** (sparsity, edges_kept/total). Similar incidents return **retrieval_provenance** (model_name, checkpoint_id, dim, timestamp). | When the checkpoint is missing, the service returns `model_available=false`; pipeline/agent use explicit rule-only fallback. Similar incidents return `available: false, reason: "model_not_run"`. |
| **GraphGPS** (`ml/models/gps_model.py`) | **Nothing in the current product.** Implemented and tested; available for **experiments or alternate training** (local message passing + global attention). | **Not** used by the pipeline, worker, or Financial Agent. **Not** loaded at inference time. Pipeline and inference only use HGT. |
| **FraudGT-style** (`ml/models/fraud_gt_style.py`) | **Elliptic benchmark pipeline only** (`ml/train_elliptic.py`, `make modal-train-elliptic`): fraud detection on the Elliptic dataset for **research/validation**. Edge-attribute attention and gating. | **Not** used for the voice/entity Independence Graph. **Not** used by the main pipeline or Financial Agent. **Does not** affect risk signals, similar incidents, or watchlists in the product. |

**When the GNN (HGT) is used:** The shared **risk scoring service** (`domain/risk_scoring_service.py`) loads the checkpoint and runs inference. If present: real scores, embeddings, PGExplainer model_subgraph (with entity IDs and evidence quality); similar incidents use pgvector RPC when available, else JSONB cosine; embedding-centroid watchlists are created. If checkpoint missing: `model_available=false`, explicit rule-only fallback; no embeddings, no model_subgraph; similar incidents and centroid watchlists are unavailable. See [docs/GNN_PRODUCT_LOOP_AUDIT.md](docs/GNN_PRODUCT_LOOP_AUDIT.md) and [docs/UPGRADE_PLAN.md](docs/UPGRADE_PLAN.md).

---

## 5. Frontend: what users can do

- **Auth:** Sign up, sign in, onboard (create household after sign-up), sign out.
- **Caregiver/Admin:** Dashboard (feed, risk chart, latest signals); **Alerts** (list with `model_available`; investigate: timeline, graph evidence, motif tags, similar incidents, feedback, Agent Trace; **Evidence-only** badge when narrative is evidence-grounded; **View ring** / **Drift warning** badges for `ring_candidate` and `drift_warning` signal types); **Sessions**; **Watchlists**; **Summaries**; **Graph** (evidence, Neo4j sync); **Ingest** (idempotent batch); **Agents** (status for all six agents with last-run summary — drift metrics, rings found, red-team pass rate; Run / Dry run per agent; View trace for any run).
- **Elder:** Elder view (simple summary, recommendation, “Share with caregiver” toggle; no raw alerts/graph).
- **Scenario Replay:** Animate scam storyline (score chart, graph, trace) from demo fixtures.
- **Demo mode:** Sidebar toggle or `NEXT_PUBLIC_DEMO_MODE=true`; app uses local fixtures, no API.

Setup: [apps/web/README.md](apps/web/README.md). Contracts: [docs/api_ui_contracts.md](docs/api_ui_contracts.md), [docs/frontend_notes.md](docs/frontend_notes.md).

---

## 6. Repo layout

```
apps/api/       FastAPI app, routers, pipeline (LangGraph), graph_state, broadcast, deps, schemas; domain (ingest idempotent, risk, risk_scoring_service, explain, similarity pgvector/JSONB, watchlist, graph, agents: financial, graph_drift, evidence_narrative, ring_discovery, continual_calibration, synthetic_redteam)
apps/worker/    Entrypoint + jobs: ingest from Supabase, run pipeline (shared scoring service), persist risk_signals/embeddings/watchlists/agent_runs, optional Neo4j sync
apps/web/       Next.js 14 dashboard (see above)
ml/             Models (HGT, GPS, FraudGT), graph builder + subgraph + time_encoding, explainers (motifs, gnn, pg), train/inference, continual, cache, Modal training
config/         settings (pipeline + ML), graph (schema, keywords), graph_policy (confidence gate)
db/             bootstrap_supabase.sql (run once), migrations 001–010 (008 = pgvector; 009 = rings/ring_members; 010 = household_calibration calibration_params), repair_households_users.sql, drop_signup_trigger.sql
scripts/        run_api.sh, run_worker.sh, start_neo4j.sh, synthetic_scenarios.py, demo_replay.py, seed_supabase_data.py, run_gnn_e2e.py, run_financial_agent_demo.py, run_migration.py, run_replay_time_to_flag.py, run_agent_cron.py (scheduled agents: drift, narrative, ring, calibration, redteam)
tests/          Pytest: config, ML, API, pipeline, worker, Modal, agents, GNN e2e, spec, strict
docs/           SUPABASE_SETUP, NEO4J_SETUP, api_ui_contracts, schema, event_packet_spec, agents, modal_training, DATA_AND_NEXT_STEPS, frontend_notes, GNN_PRODUCT_LOOP_AUDIT, UPGRADE_PLAN (refactor + GNN-driven surfaces + new agents), QUICKSTART_API, SEED_DATA, DEMO_MOMENTS
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
| GNN vs rule fallbacks, upgrade plan | [docs/GNN_PRODUCT_LOOP_AUDIT.md](docs/GNN_PRODUCT_LOOP_AUDIT.md), [docs/UPGRADE_PLAN.md](docs/UPGRADE_PLAN.md) |
| Quick start API + UI | [docs/QUICKSTART_API.md](docs/QUICKSTART_API.md) |
| Seed data, demo | [docs/SEED_DATA.md](docs/SEED_DATA.md), [docs/DEMO_MOMENTS.md](docs/DEMO_MOMENTS.md) |

---

**Tech stack (summary):** Python 3.11, FastAPI, Pydantic, Supabase (Postgres + Auth), LangGraph, PyTorch 2.2, PyTorch Geometric 2.6 (HGT, GPS, FraudGT-style), Next.js 14 (TypeScript, Tailwind, shadcn/ui, React Flow, Recharts), Modal (training), ruff, pytest.
