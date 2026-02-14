# Anchor — In-Depth Documentation

This document covers the Anchor voice-companion backend in depth: architecture, data flow, ML pipeline, agents, database schema, event format, Modal deployment, and development practices.

---

## 1. Overview and goals

**Anchor** is a backend for an edge voice companion that helps protect elders from fraud and social engineering. It does **not** capture or process raw audio in this repo; the edge device emits **structured event packets** (e.g. ASR transcripts, intents, tool calls). These are batch-uploaded (e.g. weekly), then:

1. **Ingested** into Supabase (sessions, events).
2. **Normalized** into utterances, entities, mentions, and relationships (the **Independence Graph**).
3. **Scored** by GNN/transformer-style models for fraud/risk.
4. **Explained** via subgraph attribution (GNNExplainer, PGExplainer, motifs).
5. **Surfaced** as risk signals, watchlists, and recommendations to the edge and to caregiver UIs.

The system is **read-only** for financial actions: it recommends and flags; it does not move money or execute transactions.

---

## 2. Architecture

### 2.1 High-level flow

```
Edge device → batch upload (weekly) → POST /ingest/events
       → Supabase (sessions, events)
       → Worker / pipeline: normalize → build graph → score → explain
       → risk_signals, watchlists, summaries
       → API (REST + WebSocket) → Web UI, mobile, device sync
```

### 2.2 Components

| Component | Role |
|-----------|------|
| **Edge** | Emits event packets (session_id, device_id, ts, seq, event_type, payload). No audio in repo. |
| **Backend API** | FastAPI: auth (Supabase), households, sessions, events, risk_signals, watchlists, device sync, ingest, summaries, agents. |
| **Worker** | Async jobs: ingest → graph build → train/inference. Can run as cron or one-off (`--once --household-id`). |
| **Graph** | Entity + Event nodes; time-aware edges (Δt, count, recency). See **§2.4** for the split: PyG (training/scoring), Neo4j (visualization/investigation), Supabase (source of truth). |
| **Models** | HGT baseline; GraphGPS/GPSConv; FraudGT-style edge-attribute attention. Trained on synthetic, HGB (ACM/DBLP/IMDB), or Elliptic. |
| **Explainability** | GNNExplainer, PGExplainer, SubgraphX-style; motif-based rules (urgency, sensitive info, new contact). |
| **Agents** | LangGraph orchestration; Financial Security Agent runs a 7-step playbook (ingest → detect → investigate → recommend → watchlist → escalation draft → persist). |

### 2.3 Repo layout (detailed)

```
apps/
  api/          FastAPI app (main.py), routers, pipeline, agents, broadcast (WebSocket)
  worker/       Async worker: ingest_events_batch, graph_update, risk_inference, pipeline
  web/          Next.js dashboard (alerts, agents, summaries, watchlists, etc.)
ml/             PyTorch Geometric models, training, inference, explainers
  models/       HGTBaseline, GPSRiskModel, FraudGTStyle
  graph/        builder (HeteroData from tables), subgraph, time_encoding
  explainers/   motifs, gnn_explainer, pg_explainer
  continual/    finetune_last_layer, feedback integration
  cache/        embeddings for similar-incidents
  configs/      hgt_baseline.yaml, etc.
db/             bootstrap_supabase.sql (run once), migrations/ (001–007), repair_households_users.sql, drop_signup_trigger.sql
docs/           SUPABASE_SETUP, NEO4J_SETUP, api_ui_contracts, schema, event_packet_spec, QUICKSTART_API, SEED_DATA, agents, Modal, GNN audit, demo
scripts/        run_api.sh, run_worker.sh, start_neo4j.sh, synthetic_scenarios.py, demo_replay.py, run_financial_agent_demo.py, seed_supabase_data.py, run_gnn_e2e.py, run_migration.py, run_replay_time_to_flag.py
config/         settings.py, graph.py, graph_policy.py (pipeline, ML, graph)
tests/          Pytest suite (tests/): config, ML, API, pipeline, worker, Modal, routers, spec, implementation strict, GNN e2e
```

---

## 3. Data flow

### 3.1 Event ingestion

- **Endpoint:** `POST /ingest/events` with body `{ "events": [ { session_id, device_id, ts, seq, event_type, payload_version, payload }, ... ] }`.
- **Auth:** Device-scoped; session must belong to device’s household (RLS).
- **Storage:** Rows in `events` (and session/device creation if needed). No audio; optional text in payload; `text_redacted` default true; store `text_hash` when available.

See **Event packet format** below and `docs/event_packet_spec.md`.

### 3.2 Normalization (graph build)

- **Input:** Events (and optionally utterances/entities from DB).
- **Process:** `GraphBuilder` / `build_hetero_from_tables()`:
  - Utterances from `final_asr` (and derived tables if present).
  - Entities from intents/slots (e.g. `name` → person), canonicalized.
  - Mentions link utterances/events to entities.
  - Relationships (e.g. CO_OCCURS) with weight, first_seen_at, last_seen_at.
- **Output:** PyG `HeteroData`: node types (e.g. session, utterance, entity), edge types, node features, edge attributes (time deltas, counts).

### 3.3 Risk scoring and explanation

- **Training:** `ml/train.py` (HGT/GPS/FraudGT) on synthetic, HGB, or Elliptic. Loss: focal/BCE; metrics: PR-AUC, Recall@K. Checkpoint stores metadata for inference.
- **Inference:** `ml/inference.py` loads checkpoint, builds hetero graph from household data (or in-memory), runs forward pass, optional GNNExplainer for top nodes.
- **Explainability:** Motifs (rule layer); GNNExplainer/PGExplainer/SubgraphX (model layer). Output: `explanation_json` (summary, motif_tags, timeline_snippet, subgraph, top_entities/edges).

### 3.4 Risk signals and watchlists

- **risk_signals:** severity 1–5, score, explanation, recommended_action, status (open/acknowledged/dismissed/escalated). Persisted by pipeline or Financial Security Agent.
- **watchlists:** Patterns (e.g. phone/email hashes, keywords) for edge device; priority and expiry.
- **Realtime:** New risk_signals broadcast over `WS /ws/risk_signals`.

---

## 4. Database schema (Supabase)

### 4.1 Core tables

- **households** — id, name, created_at
- **users** — id (= auth.users.id), household_id, role (elder | caregiver | admin), display_name
- **devices** — id, household_id, device_type, firmware_version, public_key, last_seen_at
- **sessions** — id, household_id, device_id, started_at, ended_at, mode (offline | online), consent_state (jsonb)
- **events** — id, session_id, device_id, ts, seq, event_type, payload (jsonb), payload_version, text_redacted, ingested_at

### 4.2 Derived / application tables

- **utterances** — session_id, ts, speaker, text, text_hash, intent, confidence
- **summaries** — household_id, session_id?, period_start/end, summary_text, summary_json
- **entities** — household_id, entity_type (person, org, phone, email, account, merchant, device, location, topic), canonical, canonical_hash, meta
- **mentions** — session_id, utterance_id?, event_id?, entity_id, ts, span, confidence
- **relationships** — household_id, src_entity_id, dst_entity_id, rel_type, weight, first_seen_at, last_seen_at, evidence
- **risk_signals** — household_id, ts, signal_type, severity, score, explanation (jsonb), recommended_action (jsonb), status
- **watchlists** — household_id, watch_type, pattern (jsonb), reason, priority, created_at, expires_at
- **device_sync_state** — device_id, last_upload_ts, last_upload_seq_by_session, last_watchlist_pull_at
- **feedback** — household_id, risk_signal_id, user_id, label (true_positive | false_positive | unsure), notes
- **agent_runs** — household_id, agent_name, started_at, ended_at, status, summary_json, step_trace (compact pipeline steps per run)
- **risk_signal_embeddings** — risk_signal_id, household_id, embedding (jsonb), dim, model_name, checkpoint_id, has_embedding, meta (migration 007); for similar-incidents and embedding-centroid watchlists; `has_embedding=false` when model did not run
- **household_calibration** — household_id, severity_threshold_adjust; for feedback-driven calibration
- **session_embeddings** — session_id, embedding (jsonb); optional session-level vectors

### 4.3 RLS

All access is **household-scoped**: users see only rows where `household_id` matches their `users.household_id`. Devices can insert events for their device and read watchlists for their household.

Schema: run **`db/bootstrap_supabase.sql`** once in Supabase SQL Editor (creates all tables, RLS, embeddings, calibration, agent_runs.step_trace, risk_signal_embeddings extended columns). If you already had an older bootstrap, run **`db/migrations/007_risk_signal_embeddings_extended.sql`** to add dim, model_name, has_embedding, etc. Repair-only: **`db/repair_households_users.sql`** (creates households/users if missing). **`db/drop_signup_trigger.sql`** removes the Auth signup trigger if signup 500s. Migrations live in `db/migrations/` (001–007).

---

### 2.4 Graph and data store boundaries

The system uses **three** representations of graph-related data. To avoid redundancy and confusion, their roles are strictly separated:

| Store | Role | When it’s used |
|-------|------|----------------|
| **Supabase** | **Source of truth** | All canonical data: sessions, events, utterances, entities, mentions, relationships, risk_signals, watchlists, feedback. Ingestion, API reads, and persistence go through Supabase. RLS enforces household scope. |
| **PyG (in-memory)** | **Training + scoring** | Pipeline builds a transient `HeteroData` graph from Supabase (or from training datasets). Used for GNN training, inference, and explainers (GNNExplainer, PGExplainer, motifs). No long-lived graph store; built per run or per household for scoring. |
| **Neo4j** | **Visualization + investigative queries** | Optional. For interactive exploration, timeline views, and ad‑hoc graph queries (e.g. “show me all paths between this entity and that session”). Not used by the ML pipeline. Populated from Supabase (or from the same normalized output that feeds PyG) when the UI or an investigation workflow needs it. |

**Summary:** Supabase is the only system of record. PyG is the compute substrate for ML. Neo4j is the human-facing graph for exploration and investigation. Training and scoring do not depend on Neo4j.

---

## 5. Event packet format (edge → backend)

- **Per-event fields:** session_id, device_id, ts, seq, event_type, payload_version, payload.
- **event_type examples:** wake, partial_asr, final_asr, intent, tool_call, tool_result, tts, error.
- **Payload variants:** e.g. final_asr (text, text_hash, lang, confidence, speaker), intent (name, slots, confidence), device_state, optional embeddings.
- **Batch:** POST /ingest/events with array of events; response: ingested count, session_ids, last_ts.

Full spec: `docs/event_packet_spec.md`.

### 5.1 Financial events (no separate schema)

The system does **not** maintain a separate transactions or accounts schema. Financial signals are treated as **event types from the edge**, like any other:

- **Examples:** `transaction_detected`, `payee_added`, `bank_alert_received`
- **Storage:** Stored in the same `events` table (session_id, device_id, ts, seq, event_type, payload).
- **Normalization:** Normalized into entities and relationships the same way as other events (e.g. merchant/person/account from payload → entities and mentions).

If **Plaid** (or similar) is added later, it is a **totally optional demo adapter** that emits the same event packets (same fields and payload shapes). No separate financial pipeline or schema.

---

## 6. ML pipeline (in depth)

### 6.1 Models

- **HGT baseline** (`ml/models/hgt_baseline.py`): Heterogeneous Graph Transformer; used for entity/session-level risk on hetero graphs.
- **GraphGPS / GPSConv** (`ml/models/gps_model.py`): For larger graphs with positional encoding.
- **FraudGT-style** (`ml/models/fraud_gt_style.py`): Edge-attribute attention; used in Elliptic pipeline (`ml/train_elliptic.py`).

### 6.2 Data sources for training

| Pipeline | Default | Real data options |
|----------|---------|-------------------|
| HGT | Synthetic (in-memory) | `--dataset hgb --hgb-name ACM` (or DBLP, IMDB, Freebase); or own data via `build_hetero_from_tables()` |
| Elliptic | — | `--dataset elliptic` (downloads via PyG) |

Synthetic data: `get_synthetic_hetero()` in `ml/train.py`; or `scripts/synthetic_scenarios.py` for scam/normal scenarios to seed events.

### 6.3 Training (local)

```bash
cd ml && python train.py --config configs/hgt_baseline.yaml --data-dir data/synthetic
# With HGB: --dataset hgb --hgb-name ACM
```

```bash
python -m ml.train_elliptic --dataset elliptic --model fraud_gt_style --data-dir data/elliptic
```

### 6.4 Inference

```bash
cd ml && python inference.py --checkpoint runs/hgt_baseline/best.pt --household-id <uuid>
```

Checkpoint carries in_channels, metadata, out_channels, hidden_channels, num_layers, heads, target_node_type so the same script works across graph types.

### 6.5 Explainers

- **Motifs** (`ml/explainers/motifs.py`): Rule-based scam-like patterns (urgency, sensitive info request, new contact, etc.).
- **GNNExplainer-style** (`ml/explainers/gnn_explainer.py`): Optimize edge mask for interpretable subgraph.
- **PGExplainer-style** (`ml/explainers/pg_explainer.py`): Parameterized edge importance.

Used in pipeline and in risk_signal detail (explanation_json, subgraph for UI).

### 6.6 Continual learning and feedback

- **ml/continual:** finetune_last_layer, feedback batch loading, threshold adjustment.
- **feedback table:** Caregiver labels (true_positive / false_positive / unsure) to improve calibration and future models.

---

## 7. Modal (serverless ML)

- **Purpose:** Run training (and optional jobs) in the cloud without managing servers.
- **Setup:** `pip install modal`, `modal setup` (or `modal token new`). Use project venv.
- **Commands:**
  - HGT: `modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml`
  - Elliptic: `modal run ml/modal_train_elliptic.py -- --dataset elliptic --model fraud_gt_style --data-dir data/elliptic`
- **Makefile:** `make modal-train`, `make modal-train-elliptic`.
- **Artifacts:** Stored on Modal Volume `anchor-runs` (e.g. runs/hgt_baseline/best.pt, runs/elliptic/metrics.json). Download via Modal CLI or a one-off function.
- **Secrets:** Use `modal.Secret` for Supabase/API keys if a job needs them.
- **GPU:** Enable by uncommenting `gpu="T4"` (or similar) in the Modal function decorators.

Details: `docs/modal_training.md`, `docs/DATA_AND_NEXT_STEPS.md`.

---

## 8. Financial Security Agent

### 8.1 Role

Read-only financial protection: recommend and flag; never move money. Runs a 7-step playbook.

### 8.2 Playbook steps

1. **Ingest & normalize** — Recent events for household (configurable window, default 7 days) or pre-filled state; normalize via GraphBuilder.
2. **Detect risk** — Motif/rule layer (scam-like patterns) + optional GNN scoring; output risk score, severity 1–5, uncertainty.
3. **Investigation package** — motif_tags, timeline_snippet, evidence subgraph, “what changed” summary; uses explainers when available.
4. **Recommendations** — recommended_action with checklist (e.g. call back via saved number, don’t share OTP, enable bank alerts, verify payee).
5. **Watchlist synthesis** — Turn risk into watchlist items (hashes, keywords) for edge; insert into `watchlists` when not dry_run and consent allows.
6. **Consent-gated escalation** — If severity ≥ threshold and consent share_with_caregiver: draft escalation message (stored, not sent). Otherwise no escalation beyond elder scope.
7. **Persist + notify** — Insert/update `risk_signals`; broadcast on `/ws/risk_signals` when run via API with persist.

### 8.3 API and testing

- **POST /agents/financial/run** — Body: household_id?, time_window_days?, dry_run?, use_demo_events?. dry_run returns risk_signals and watchlists without writing.
- **GET /agents/status** — List agents, last run time/status/summary.
- **GET /agents/financial/trace?run_id=** — Trace for a given run.
- **GET /agents/financial/demo** — Demo input/output (no auth, no DB write).

**One-command demo:** `PYTHONPATH=".:apps/api" python3 scripts/demo_replay.py` — seeds scenario, runs pipeline, writes risk_chart.json, explanation_subgraph.json, agent_trace.json, scenario_replay.json; optional `--ui --launch-ui` to update fixtures and start web in demo mode.  
**Local script (no API):** `PYTHONPATH=".:apps/api" python3 scripts/run_financial_agent_demo.py`  
**Unit tests:** `pytest tests/test_financial_agent.py -v`

Full description: `docs/agents.md`.

---

## 9. API contracts (summary)

- **Auth:** Supabase Auth; JWT in `Authorization: Bearer <token>`. Roles: elder, caregiver, admin. Household from `GET /households/me`. After sign-up: `POST /households/onboard` to create household and link user (recommended; see docs/SUPABASE_SETUP.md).
- **REST:** households/me, households/onboard, sessions, sessions/{id}/events, risk_signals (list + detail + similar + feedback), watchlists, device/sync, ingest/events, summaries, agents/financial/run, agents/status, agents/financial/trace.
- **Realtime:** `WS /ws/risk_signals` — new risk_signal payloads as they are created.
- **JSON shapes:** risk signal card/detail (with subgraph), weekly summary, watchlist item — see `docs/api_ui_contracts.md` and `docs/frontend_notes.md`.

---

## 10. Development guide

### 10.1 Environment

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[ml]"
cp .env.example .env   # set SUPABASE_* if using API/worker with DB
```

### 10.2 Running locally

- **Supabase (optional):** `supabase init && supabase start && supabase db push`; set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY in .env.
- **API:** `./scripts/run_api.sh` → http://127.0.0.1:8000 (docs at /docs).
- **Worker:** `./scripts/run_worker.sh`; one-off: `./scripts/run_worker.sh --once --household-id <uuid>`.
- **Synthetic data:** `python scripts/synthetic_scenarios.py --household-id <uuid> --output db/seed_events.json`.

### 10.3 Testing

```bash
make test   # ruff + pytest
# or: ruff check apps ml tests && pytest tests -v
```

- **tests/README.md** — Full map of test files (config, ML, API, pipeline, worker, Modal, routers, spec tests).
- **Spec tests** (`*_spec.py`) — Encode documented behavior (formulas, thresholds); changes to behavior may require updating specs.

### 10.4 Linting

```bash
make lint   # ruff check apps ml tests
```

### 10.5 Dependencies

- **pyproject.toml:** Core deps (FastAPI, Supabase, LangGraph, PyTorch, PyG, Modal, pytest, ruff). Optional `[ml]`: torch-scatter, torch-sparse, torch-cluster for full PyG support.

---

## 11. Document index

| Document | Content |
|----------|---------|
| **README.md** | Quick start, repo layout, Modal, run instructions, testing, Makefile, tech stack |
| **README_EXTENDED.md** | This file — architecture, graph/store boundaries (§2.4), data flow, schema, ML, agents, Modal, dev guide |
| **docs/api_ui_contracts.md** | REST and WebSocket contracts, JSON schemas for UI |
| **docs/agents.md** | Financial Security Agent playbook, API, testing, safety |
| **docs/schema.md** | Core and derived tables (including embeddings, calibration), RLS |
| **docs/event_packet_spec.md** | Event fields and payload variants, batch ingest |
| **docs/modal_training.md** | Modal setup, HGT/Elliptic commands, Volume, GPU |
| **docs/DATA_AND_NEXT_STEPS.md** | Real data (HGB, Elliptic), training→eval→inference, checkpoint download |
| **docs/frontend_notes.md** | Data objects and query params for frontend |
| **docs/DEMO_MOMENTS.md** | Demo narrative (temporal, subgraph, similar incidents, HITL, edge, Elliptic) |
| **docs/SUPABASE_SETUP.md** | Full Supabase setup: bootstrap SQL, .env, auth, link user (onboard or trigger), repair/drop_signup |
| **docs/supabase-setup.md** | Connecting Supabase: keys, env (cross-ref to SUPABASE_SETUP.md for full setup) |
| **docs/NEO4J_SETUP.md** | Optional Neo4j: Docker, start_neo4j.sh, API env, Graph view sync, /graph/neo4j-status |
| **docs/QUICKSTART_API.md** | Run API + frontend in two terminals |
| **docs/SEED_DATA.md** | seed_supabase_data.py: full demo seed, options, pipeline follow-up |
| **docs/GNN_PRODUCT_LOOP_AUDIT.md** | Audit: where GNN is used vs rule-based fallbacks |
| **tests/README.md** | Test layout, spec-based tests, strict implementation tests, GNN e2e |

---

*Anchor — Voice companion backend with Independence Graph, GNN risk scoring, and explainability.*
