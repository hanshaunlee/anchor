# Anchor — Codebase Reference (Extended)

This document is a **file-by-file and module-by-module** reference of the Anchor repository. It is generated from a full review of the codebase so that any developer can understand where every piece of behavior lives and how components connect.

---

## 1. Overview and goals

**Anchor** is a backend for an edge voice companion that helps protect elders from fraud and social engineering.

- **No raw audio** in this repo: the edge device emits **structured event packets** (ASR transcripts, intents, tool calls, financial events).
- Events are **batch-uploaded** (e.g. weekly) via `POST /ingest/events`.
- The backend **ingests** into Supabase, **normalizes** into an **Independence Graph** (utterances, entities, mentions, relationships), **scores** risk with GNN/rule layers, **explains** via motifs and optional GNN explainers, and **surfaces** risk signals, watchlists, and recommendations to the edge and caregiver UIs.
- The system is **read-only** for money: it recommends and flags; it does not execute financial transactions.

---

## 2. Repository structure (every directory and file)

### 2.1 Root

| File | Purpose |
|------|---------|
| `README.md` | Quick start, repo layout, Modal, run instructions, testing, Makefile, tech stack, further reading. |
| `README_EXTENDED.md` | This file — full codebase reference. |
| `pyproject.toml` | Package definition: `anchor`, Python ≥3.11, deps (FastAPI, Supabase, LangGraph, PyTorch, PyG, Modal, pytest, ruff). Optional `[ml]`: torch-scatter, torch-sparse, torch-cluster. Optional `[db]`: psycopg2-binary. Packages: `apps*`, `ml*`, `db*`. Ruff line-length 100, py311. Pytest: asyncio_mode=auto, testpaths = tests, apps/api/tests, apps/worker/tests, ml/tests. |
| `Makefile` | Targets: install, test, lint, migrate, dev-api, dev-worker, synth, train, modal-train, modal-train-elliptic. |
| `modal_app.py` | Modal app entrypoint: `modal.App("anchor")`, `hello()` function; run via `modal run modal_app.py` or `modal deploy modal_app.py`. Actual ML training uses `ml/modal_train.py` and `ml/modal_train_elliptic.py`. |

### 2.2 `apps/`

#### 2.2.1 `apps/api/`

**Entrypoint and config**

- **`api/main.py`**  
  FastAPI app: title "Anchor API", CORS enabled. Routers: households, graph, sessions, risk_signals, watchlists, device, ingest, summaries, agents. WebSocket `GET /ws/risk_signals`: add/remove subscriber, push new risk_signal payloads. Routes: `GET /` (links to docs, health, demo), `GET /health` → `{"status":"ok"}`. Lifespan: no-op.

- **`api/config.py`**  
  Pydantic Settings: `supabase_url`, `supabase_service_role_key`, `database_url`, `jwt_secret`, `neo4j_uri`, `neo4j_user`, `neo4j_password`. Loads from `.env`; `extra = "ignore"`.

- **`api/deps.py`**  
  Dependencies: `get_supabase()` — creates Supabase client from config, raises 503 if URL/key missing. `get_current_user_id(credentials, supabase)` — validates JWT via `supabase.auth.get_user`, returns user id or None. `require_user(user_id)` — raises 401 if None. Uses `HTTPBearer` and `OAuth2PasswordBearer` with `auto_error=False`.

**State and pipeline**

- **`api/graph_state.py`**  
  `AnchorState`: Pydantic model for LangGraph state (household_id, time_range_*, ingested_events, session_ids, normalized, utterances, entities, mentions, relationships, graph_updated, risk_scores, explanations, consent_*, watchlists, escalation_draft, persisted, needs_review, severity_threshold, consent_state, time_to_flag, logs). `append_log(state, msg)` appends to `state["logs"]`.

- **`api/pipeline.py`**  
  LangGraph pipeline: `ingest` → `normalize` (deterministic: events sorted by session, seq) → `graph_update` → `financial_security_agent` → `risk_score` (calls **domain.risk_scoring_service.score_risk**; explicit rule-only fallback when `model_available=false`) → `explain` (motifs + model_subgraph with model_evidence_quality) → `consent_gate` → `watchlist` → `escalation_draft` → `persist`. Key: `risk_score_inference` uses shared scoring service; `generate_explanations` copies model_evidence_quality; `_embedding_centroid_watchlist` (pattern with cosine_threshold, provenance). `build_graph(checkpointer)`, `run_pipeline(...)`.

- **`api/broadcast.py`**  
  In-memory set of WebSocket subscribers. `add_subscriber(ws)`, `remove_subscriber(ws)`, `broadcast_risk_signal(payload)` — sends JSON to all subscribers (used when new risk_signal is persisted via API/agent).

**Routers**

- **`api/routers/households.py`**  
  Prefix `/households`. `POST /onboard` — create household and users row for authenticated user (idempotent if already onboarded); body `OnboardRequest(display_name?, household_name?)`. `GET /me` — return `HouseholdMe` (id, name, role, display_name) for current user; 404 if not onboarded.

- **`api/routers/graph.py`**  
  Prefix `/graph`. `GET /evidence` — build evidence subgraph from household events (GraphBuilder), return `RiskSignalDetailSubgraph` (nodes/edges). `POST /sync-neo4j` — mirror that subgraph to Neo4j (no-op if not configured). `GET /neo4j-status` — `{enabled, browser_url?, connect_url?, password?}` for UI.

- **`api/routers/sessions.py`**  
  Session list and session events (see docs/api_ui_contracts.md for paths).

- **`api/routers/risk_signals.py`**  
  List risk signals, get by id, feedback, similar incidents (see api_ui_contracts).

- **`api/routers/watchlists.py`**  
  List watchlists for household.

- **`api/routers/device.py`**  
  `POST /device/sync` — device heartbeat, returns watchlist delta and upload pointers.

- **`api/routers/ingest.py`**  
  `POST /ingest/events` — validate sessions belong to household, insert events; uses domain `ingest_service.ingest_events`.

- **`api/routers/summaries.py`**  
  List weekly/session summaries.

- **`api/routers/agents.py`**  
  Prefix `/agents`. Financial: `POST /financial/run`, `GET /financial/demo`, `GET /financial/trace?run_id=`. Other agents: `POST /drift/run`, `POST /narrative/run`, `POST /ring/run`, `POST /calibration/run`, `POST /redteam/run` (each with dry_run; persist to agent_runs with step_trace). `GET /status` — last run for all six agents (financial_security, graph_drift, evidence_narrative, ring_discovery, continual_calibration, synthetic_redteam). `GET /trace?run_id=&agent_name=` — generic trace for any agent.

**Schemas**

- **`api/schemas.py`**  
  Pydantic models: enums; event packet; HouseholdMe, OnboardRequest; sessions/events; **RiskScoreItem, RiskScoringResponse** (single risk scoring contract); RiskSignalCard (model_available), RiskSignalDetail, SubgraphNode/Edge, RiskSignalDetailSubgraph, FeedbackSubmit, RiskSignalListResponse; **EmbeddingCentroidPattern, EmbeddingCentroidProvenance**; WatchlistItem (model_available); DeviceSyncRequest/Response; SimilarIncident, **RetrievalProvenance**, SimilarIncidentsResponse (retrieval_provenance); WeeklySummary. Contract of record for API and event_packet_spec.

**Neo4j**

- **`api/neo4j_sync.py`**  
  `_driver()` — lazy GraphDatabase.driver from api.config (neo4j_uri, user, password); None if unset or neo4j not installed. `sync_evidence_graph_to_neo4j(household_id, entities, relationships)` — MERGE entities and relationships into Neo4j; clears household subgraph first. `neo4j_enabled()` — True when driver is non-None.

**Domain layer**

- **`domain/__init__.py`**  
  Package marker.

- **`domain/ingest_service.py`**  
  `get_household_id(supabase, user_id)`. `ingest_events(...)` — validate sessions in household, **upsert** event rows with `on_conflict="session_id,seq"` (idempotent), return IngestEventsResponse.

- **`domain/risk_service.py`**  
  Risk signal list (RiskSignalCard with **model_available** from explanation), detail (explanation with **model_available** when missing), submit_feedback, calibration update.

- **`domain/explain_service.py`**  
  `build_subgraph_from_explanation(explanation)` — build RiskSignalDetailSubgraph from explanation subgraph/model_subgraph. `get_similar_incidents(signal_id, household_id, supabase, top_k)` — delegates to similarity_service.

- **`domain/similarity_service.py`**  
  Similar incidents: tries **pgvector RPC** `similar_incidents_by_vector` (migration 008) when available; fallback to JSONB + Python cosine. Returns SimilarIncidentsResponse with **retrieval_provenance** (model_name, checkpoint_id, embedding_dim, timestamp) when available; `available=false, reason="model_not_run"` when no real embedding.

- **`domain/watchlist_service.py`**  
  Watchlist listing (**model_available=True** for watch_type=embedding_centroid); device sync.

- **`domain/graph_service.py`**  
  `normalize_events` — build utterances/entities/mentions/relationships from events (GraphBuilder); **deterministic** (events sorted by ts, seq per session).

- **`domain/risk_scoring_service.py`**  
  **Single risk scoring contract:** `score_risk(household_id, sessions, utterances, entities, mentions, relationships, devices?, events?, checkpoint_path?, explanation_score_min?)` → `RiskScoringResponse(model_available, scores, fallback_used?)`. Used by pipeline, worker, and Financial Security Agent. No silent placeholders; on failure returns `model_available=false`, empty scores. When model runs: PGExplainer inline (stable entity IDs in model_subgraph), model_evidence_quality (sparsity, edges_kept/total).

- **`domain/agents/__init__.py`**  
  Package marker.

- **`domain/agents/financial_security_agent.py`**  
  Financial Security Agent: **DEMO_EVENTS**, **get_demo_events()**; **_ingest_events**; **normalize_events** (graph_service); **_detect_risk_patterns** (calls **domain.risk_scoring_service.score_risk** for GNN path; motif + model_available; combined 0.6*rule + 0.4*model when model ran); **_watchlist_synthesis**; **run_financial_security_playbook(...)** — full playbook, persists risk_signals/watchlists/agent_runs.

- **`domain/agents/graph_drift_agent.py`**  
  **run_graph_drift_agent(household_id, supabase, dry_run, tau)** — embedding distribution shift; if shift > τ opens risk_signal type `drift_warning`; returns step_trace, summary_json, status.

- **`domain/agents/evidence_narrative_agent.py`**  
  **run_evidence_narrative_agent(household_id, risk_signal_id?, supabase, dry_run)** — model_subgraph + motifs → caregiver summary; stores in risk_signals.explanation.summary.

- **`domain/agents/ring_discovery_agent.py`**  
  **run_ring_discovery_agent(household_id, supabase, neo4j_available, dry_run)** — Neo4j GDS node similarity; flag clusters; stub when Neo4j unavailable.

- **`domain/agents/continual_calibration_agent.py`**  
  **run_continual_calibration_agent(household_id, supabase, dry_run)** — update household calibration from feedback; calibration report.

- **`domain/agents/synthetic_redteam_agent.py`**  
  **run_synthetic_redteam_agent(household_id, supabase, dry_run)** — generate scam variants; validate Similar Incidents + centroid watchlists (regression).

**API agents (LangGraph wiring)**

- **`api/agents/__init__.py`**  
  Package marker.

- **`api/agents/financial_agent.py`**  
  Thin wrapper: imports domain `run_financial_security_playbook` for use from pipeline or routers.

#### 2.2.2 `apps/worker/`

- **`main.py`**  
  Entrypoint: `python -m worker.main`. Parses `--household-id`, `--once`. If `--once` and household-id: calls `worker.jobs.run_pipeline(supabase, household_id)`. Otherwise logs "Worker idle". Adds ROOT and apps/api to sys.path for imports.

- **`worker/__init__.py`**  
  Package marker.

- **`worker/jobs.py`**  
  **ingest_events_batch** — fetch events. **run_graph_builder** — GraphBuilder per session; optional Neo4j sync. **run_risk_inference(household_id, graph_data, checkpoint_path)** — calls **domain.risk_scoring_service.score_risk**; returns explicit rule-only fallback when model unavailable. **_check_embedding_centroid_watchlists**. **run_pipeline** — fetch events, call **api.pipeline.run_pipeline**; persist risk_signals, risk_signal_embeddings (only when model ran), watchlists, agent_runs.step_trace.

#### 2.2.3 `apps/web/`

Next.js 14 App Router dashboard. Key paths:

- **`src/app/page.tsx`** — Landing.
- **`src/app/(auth)/login/page.tsx`**, **signup/**, **onboard/**, **logout/** — Auth and onboarding.
- **`src/app/(dashboard)/dashboard/page.tsx`** — Caregiver home.
- **`src/app/(dashboard)/alerts/page.tsx`**, **alerts/[id]/page.tsx`**, **alert-detail-content.tsx** — Risk signals list and detail (timeline, graph, similar incidents, feedback).
- **`src/app/(dashboard)/sessions/page.tsx`**, **sessions/[id]/** — Sessions and events.
- **`src/app/(dashboard)/watchlists/page.tsx`** — Watchlists.
- **`src/app/(dashboard)/summaries/page.tsx`** — Weekly summaries.
- **`src/app/(dashboard)/graph/page.tsx`** — Graph view (evidence subgraph, Sync to Neo4j, Open in Neo4j Browser).
- **`src/app/(dashboard)/ingest/page.tsx`** — Event ingest.
- **`src/app/(dashboard)/agents/page.tsx`** — Agent center (dry run, trace).
- **`src/app/(dashboard)/elder/page.tsx`** — Elder view.
- **`src/app/(dashboard)/replay/page.tsx`** — Scenario replay (score chart, graph, trace).
- **`src/lib/api/client.ts`**, **schemas.ts** — API client and types.
- **`src/components/dashboard-nav.tsx`**, **graph-evidence.tsx** — Nav and graph viz.
- **`public/fixtures/`** — Demo mode JSON (risk_signals, sessions, scenario_replay, etc.).

See **apps/web/README.md** for stack, env, routes, and demo mode.

### 2.3 `ml/`

**Training and inference**

- **`ml/train.py`**  
  CLI: train HGT (or GPS/FraudGT) on synthetic or HGB data. **focal_loss**, **_hetero_metadata**, **get_hgb_hetero(data_dir, name)**, **get_synthetic_hetero(data_dir)** (uses same build_hetero_from_tables schema as inference), **_get_hetero_labels**, train/eval loop; checkpoint saves in_channels, metadata, model_state, hidden_channels, out_channels, num_layers, heads, target_node_type. Config from ml.config.get_train_config.

- **`ml/inference.py`**  
  **load_model(checkpoint_path, device)** — load HGT from checkpoint, return (model, target_node_type). **run_inference(model, data, device, target_node_type, explain_node_idx, return_embeddings)** — forward, softmax, risk list with optional embedding per node; optional GNNExplainer for explain_node_idx. CLI: load checkpoint, build graph from tables or synthetic, run inference, optional explain, output JSON.

- **`ml/train_elliptic.py`**  
  Elliptic dataset (or synthetic fallback); FraudGT-style model; train/eval; metrics and embedding plot.

- **`ml/config.py`**  
  **get_train_config()** — load YAML (e.g. configs/hgt_baseline.yaml) for training.

**Models**

- **`ml/models/hgt_baseline.py`**  
  HGTBaseline: HeteroConv (HGTConv), linear out per node type; forward returns logits; **forward_hetero_data_with_hidden** returns (logits, hidden_dict) for explainers and embeddings.

- **`ml/models/gps_model.py`**  
  GraphGPS / GPSConv for larger graphs with positional encoding.

- **`ml/models/fraud_gt_style.py`**  
  FraudGT-style edge-attribute attention; used in Elliptic pipeline.

**Graph**

- **`ml/graph/builder.py`**  
  **GraphBuilder(household_id)** — utterances list, entities dict keyed by (entity_type, canonical_hash), mentions, relationships (src, dst, rel_type, weight, first_seen_at, last_seen_at, evidence). **_get_or_create_entity**. **process_events(events, session_id, device_id)** — final_asr → utterances; intent → intents + link to last utterance; transaction_detected/payee_added/bank_alert_received → entities/mentions; intent slots (slot_to_entity) → person/phone/email entities; co-occurrence within window_sec → CO_OCCURS relationships; watchlist_hit → TRIGGERED relationship. Respects **config.graph_policy.allow_graph_mutation_for_event** (ASR/intent confidence gate). **get_utterance_list**, **get_entity_list**, **get_mention_list**, **get_relationship_list**. **build_hetero_from_tables(household_id, sessions, utterances, entities, mentions, relationships, devices)** — builds PyG HeteroData with node types (session, utterance, entity, etc.) and edge types from config.graph; time encoding; same schema as training get_synthetic_hetero.

- **`ml/graph/subgraph.py`**  
  k-hop subgraph extraction (extract_k_hop, time_window) for evidence and explainers.

- **`ml/graph/time_encoding.py`**  
  Sinusoidal time encoding for node/edge features (TGAT-style).

**Explainers**

- **`ml/explainers/motifs.py`**  
  **extract_motifs(utterances, mentions, entities, relationships, events, entity_id_to_canonical, urgency_topics?, sensitive_intents?)** — returns (motif_tags, timeline_snippet). Motifs: new contact + urgency topic, bursty contact, device switching, contact→sensitive intent→payee cascade. Keywords from config.graph (urgency_topics, sensitive_intents).

- **`ml/explainers/gnn_explainer.py`**  
  GNNExplainer-style: optimize edge mask for interpretable subgraph; **explain_node_gnn_explainer_style**.

- **`ml/explainers/pg_explainer.py`**  
  PGExplainerStyle: parameterized edge importance; **explain_with_pg** returns top_edges and minimal_subgraph_node_ids.

**Continual and cache**

- **`ml/continual/finetune_last_layer.py`**  
  Load feedback batch, threshold adjustment; **finetune_last_layer_stub** (no actual training in stub).

- **`ml/cache/embeddings.py`**  
  EmbeddingCache for session/risk_signal embeddings; get_similar_sessions (cosine).

**Modal**

- **`ml/modal_train.py`**  
  Modal app: image, volume for runs; **main** runs ml.train with passed args (config, data-dir, epochs, device); **download_checkpoint** to pull best.pt to local.

- **`ml/modal_train_elliptic.py`**  
  Modal app for Elliptic training (dataset elliptic, model fraud_gt_style, output runs/elliptic).

**Config**

- **`ml/configs/hgt_baseline.yaml`**  
  HGT training config (hidden, layers, heads, etc.).

### 2.4 `config/`

- **`config/settings.py`**  
  **PipelineSettings** (env_prefix ANCHOR_): risk_score_threshold, explanation_score_min, watchlist_score_min, escalation_score_min, persist_score_min, severity_threshold, timeline_snippet_max, consent_share_key, consent_watchlist_key, default_consent_*, asr_confidence_min_for_graph. **MLSettings** (ANCHOR_ML_): embedding_dim, risk_inference_entity_cap, calibration_*_step/cap/floor, model_version_tag, checkpoint_path, explainer_epochs. **get_pipeline_settings()**, **get_ml_settings()** (lru_cache). **get_ml_config_path()** (env ANCHOR_ML_CONFIG). **_load_yaml**.

- **`config/graph.py`**  
  **get_graph_config()** (lru_cache): node_types, edge_types, entity_type_map, slot_to_entity, event_types, speaker_roles, base_feature_dims, time_encoding_dim, edge_attr_dim, co_occurrence_window_sec, person_ids, urgency_topics, sensitive_intents. Can override via YAML (ANCHOR_GRAPH_CONFIG).

- **`config/graph_policy.py`**  
  **allow_graph_mutation_for_event(event, confidence_min)** — True only when event is not confidence-gated or payload.confidence >= threshold. final_asr and intent are gated; others allowed. Uses config.settings.asr_confidence_min_for_graph.

### 2.5 `db/`

- **`db/bootstrap_supabase.sql`**  
  Full schema for new project: enums (user_role, session_mode, speaker_type, entity_type_enum, risk_signal_status, feedback_label); tables households, users, devices, sessions, events, utterances, summaries, entities, mentions, relationships, risk_signals, watchlists, device_sync_state, feedback, session_embeddings, risk_signal_embeddings (with dim, model_name, checkpoint_id, has_embedding, meta), household_calibration, agent_runs (with step_trace); indexes; RLS and policies; helper functions (user_household_id, user_role). Run once in Supabase SQL Editor.

- **`db/migrations/001_initial_schema.sql`** … **008_pgvector_embeddings.sql`**  
  Incremental migrations. 007: extended risk_signal_embeddings (dim, model_name, has_embedding, etc.). **008**: optional pgvector — `CREATE EXTENSION vector`; `embedding_vector vector(32)`; IVFFLAT cosine index; RPC **similar_incidents_by_vector**; backfill from JSONB where dim=32. When extension unavailable, app uses JSONB + Python cosine.

- **`db/repair_households_users.sql`**  
  Idempotent: create user_role enum, households, users if missing; RLS and policies for households/users. Use when trigger or partial bootstrap left tables missing.

- **`db/drop_signup_trigger.sql`**  
  Drop trigger on_auth_user_created and function handle_new_user so signup no longer runs trigger (fix 500 on signup); frontend uses POST /households/onboard instead.

### 2.6 `scripts/`

- **`run_api.sh`** — From repo root: set PYTHONPATH to apps/api, run uvicorn api.main:app (reload on apps/api and config).
- **`run_worker.sh`** — From repo root: set PYTHONPATH to root, apps, apps/api; run python -m worker.main.
- **`start_neo4j.sh`** — Docker: start Neo4j container (default password neo4j123); NEO4J_IMAGE, NEO4J_CONTAINER, NEO4J_PASSWORD env.
- **`synthetic_scenarios.py`** — Generate normal + scam scenario events; --household-id, --output.
- **`demo_replay.py`** — One-command demo: run pipeline on demo events, write risk_chart.json, explanation_subgraph.json, agent_trace.json, scenario_replay.json to demo_out/; --ui --launch-ui to update fixtures and open dashboard.
- **`run_financial_agent_demo.py`** — Run financial playbook on demo events locally (no API).
- **`seed_supabase_data.py`** — Full seed: households, users, devices, sessions, events, utterances, entities, mentions, relationships, summaries, risk_signals, watchlists, feedback, agent_runs; --dry-run, --output-json, --household-id, --user-id, SEED_USER_ID.
- **`run_gnn_e2e.py`** — Train HGT if needed, run pipeline with checkpoint, assert embeddings and model_subgraph; --skip-train, --train.
- **`run_migration.py`** — Run migrations (DATABASE_URL).
- **`run_replay_time_to_flag.py`** — Replay events, compute time_to_flag metrics.

### 2.7 `tests/`

Pytest suite under **tests/** (and optionally apps/api/tests, apps/worker/tests, ml/tests per pyproject). Key files:

- **conftest.py** — Shared fixtures.
- **test_config_*.py** — Pipeline/ML/graph settings, init, graph config.
- **test_ml_*.py** — Config, time_encoding, subgraph, builder, builder_units, builder_spec, models, fraud_gt_style, train, train_spec, train_elliptic, inference, cache, continual, motifs, explainers_gnn, explainers_pg.
- **test_api_*.py** — main, deps, config, routers, routers_agents, routers_extra, routers_risk_signals_detail, routers_device, routers_ingest, routers_sessions_events.
- **test_pipeline*.py** — pipeline, pipeline_nodes, pipeline_build_graph, pipeline_financial_node, pipeline_spec.
- **test_financial_agent.py** — Playbook, consent, demo events.
- **test_graph_state.py**, **test_schemas.py**, **test_schemas_extended.py** — State and Pydantic schemas.
- **test_broadcast.py**, **test_broadcast_multi.py** — WebSocket subscribers.
- **test_worker_jobs.py**, **test_worker_main.py**, **test_worker_jobs_extended.py** — Jobs and main.
- **test_modal.py** — modal_app structure, modal_train entrypoints.
- **test_scripts_demo.py** — demo_replay contract.
- **test_synthetic_scenarios.py** — synthetic_scenarios helpers.
- **test_implementation_strict.py** — Strict behavior and failure paths (ingest prefill, append_log, deps 503/401, focal gamma=0, build_hetero entity_to_idx, financial node exception, should_review, watchlist_hit TRIGGERED).
- **test_gnn_product_loop.py** — GNN audit assertions.
- **test_graph_policy.py** — allow_graph_mutation_for_event.
- **test_gnn_e2e.py** — E2E pipeline with checkpoint (embeddings, similar, model_subgraph).

See **tests/README.md** for full layout and spec/strict test descriptions.

### 2.8 `docs/`

- **SUPABASE_SETUP.md** — New project, bootstrap SQL, .env, auth, link user (Option C onboard, A manual, B trigger), troubleshooting 500 signup, repair/drop_signup, verify.
- **NEO4J_SETUP.md** — Optional Neo4j: Docker, start_neo4j.sh, API env, Graph view sync, /graph/neo4j-status, /graph/sync-neo4j.
- **api_ui_contracts.md** — Auth, REST table, WebSocket, JSON schemas (risk card/detail, summary, watchlist), UI inputs/outputs summary.
- **schema.md** — Core and derived tables, RLS; risk_signal_embeddings extended (dim, model_name, has_embedding, meta).
- **event_packet_spec.md** — Event fields, payload variants (final_asr, intent, device_state, financial), batch ingest.
- **agents.md** — Financial Security Agent playbook, API, consent, safety.
- **QUICKSTART_API.md** — Run API and frontend in two terminals.
- **SEED_DATA.md** — seed_supabase_data.py, options, pipeline follow-up.
- **modal_training.md** — Modal HGT/Elliptic commands, Volume, GPU.
- **DATA_AND_NEXT_STEPS.md** — Real data (HGB, Elliptic), training→eval→inference, checkpoint download.
- **frontend_notes.md** — Data objects (HouseholdMe, Session, Event, RiskSignal, etc.) and endpoints for UI.
- **DEMO_MOMENTS.md** — Demo narrative (temporal, subgraph, similar incidents, HITL, edge, Elliptic).
- **GNN_PRODUCT_LOOP_AUDIT.md** — Where GNN is used vs rule fallbacks; “delete the GNN” litmus test.
- **UPGRADE_PLAN.md** — Refactor deliverables (single risk scoring, idempotent ingest, model_available); GNN-driven surfaces (pgvector, embedding-centroid, evidence subgraph); new agents; operational (status, trace).
- **supabase-setup.md** — Keys and env; cross-ref to SUPABASE_SETUP for full setup.

---

## 3. Data flow (end-to-end)

1. **Edge** → batch events → **POST /ingest/events** (device/auth); **ingest_service.ingest_events** validates sessions in household, inserts into `events`.
2. **Worker** (or on-demand): **jobs.run_pipeline** fetches events, runs **api.pipeline.run_pipeline**. **risk_score** uses **domain.risk_scoring_service.score_risk** (shared contract); normalize is deterministic; explain includes model_evidence_quality and stable entity IDs.
3. **Persistence**: risk_signals, risk_signal_embeddings (only when model ran; optional pgvector column per migration 008), watchlists, agent_runs.step_trace; optional Neo4j sync.
4. **API** serves: households, sessions, risk_signals (list with model_available, detail, feedback, similar with retrieval_provenance; pgvector RPC or JSONB cosine), watchlists (model_available for embedding_centroid), device/sync, **ingest (idempotent upsert)**, summaries, **agents** (financial run/demo/trace; drift, narrative, ring, calibration, redteam run; status; **GET /agents/trace?run_id=&agent_name=**); graph/evidence, graph/sync-neo4j, graph/neo4j-status.
5. **WebSocket** **/ws/risk_signals** pushes new risk_signal payloads when **broadcast_risk_signal** is called (after persist or agent run).
6. **Financial Agent** (on-demand or inside pipeline): **run_financial_security_playbook** — ingest (DB or pre-filled) → normalize → detect (rule + optional GNN) → investigation bundle → recommendations → watchlist synthesis → escalation draft → persist + broadcast.

---

## 4. Graph and data store boundaries

| Store | Role |
|-------|------|
| **Supabase** | Source of truth: households, users, devices, sessions, events, utterances, entities, mentions, relationships, risk_signals, watchlists, feedback, agent_runs, risk_signal_embeddings, household_calibration, session_embeddings. RLS: household-scoped. |
| **PyG (in-memory)** | Built per run from Supabase (or training data) via **build_hetero_from_tables**. Used for GNN training, inference, and explainers. No persistent graph DB. |
| **Neo4j** | Optional. Evidence subgraph (entities + relationships) mirrored from API/worker for visualization and Cypher. Not used by ML. |

---

## 5. Configuration summary

- **API**: `.env` or `apps/api/.env` — SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY; optional NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.
- **Pipeline/ML**: `config.settings` — PipelineSettings (ANCHOR_*), MLSettings (ANCHOR_ML_*); override via env. Graph: config.graph (ANCHOR_GRAPH_CONFIG YAML).
- **Graph policy**: config.graph_policy uses PipelineSettings.asr_confidence_min_for_graph for final_asr/intent confidence gate.

---

## 6. Document index

| Document | Content |
|----------|---------|
| **README.md** | Quick start, layout, Modal, run steps, testing, Makefile, tech stack, further reading. |
| **README_EXTENDED.md** | This file — full codebase reference. |
| **docs/SUPABASE_SETUP.md** | Bootstrap, .env, auth, onboard, repair, drop_signup. |
| **docs/NEO4J_SETUP.md** | Neo4j Docker, API env, Graph view. |
| **docs/api_ui_contracts.md** | REST, WebSocket, JSON schemas. |
| **docs/schema.md** | Tables, RLS, embeddings. |
| **docs/event_packet_spec.md** | Event format, payloads. |
| **docs/agents.md** | Financial Agent. |
| **docs/QUICKSTART_API.md** | API + frontend two terminals. |
| **docs/SEED_DATA.md** | seed_supabase_data.py. |
| **docs/modal_training.md** | Modal HGT/Elliptic. |
| **docs/DATA_AND_NEXT_STEPS.md** | Real data, checkpoint. |
| **docs/frontend_notes.md** | UI data objects. |
| **docs/DEMO_MOMENTS.md** | Demo narrative. |
| **docs/GNN_PRODUCT_LOOP_AUDIT.md** | GNN vs rules. |
| **docs/UPGRADE_PLAN.md** | Refactor + GNN surfaces + new agents. |
| **docs/supabase-setup.md** | Keys, env. |
| **tests/README.md** | Test layout, spec, strict. |
| **apps/web/README.md** | Web stack, routes, demo mode. |

---

*Anchor — Voice companion backend with Independence Graph, GNN risk scoring, and explainability.*
