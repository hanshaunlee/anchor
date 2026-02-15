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
  FastAPI app: title "Anchor API", CORS enabled. Routers: households, graph, sessions, alerts, risk_signals, protection, explain, watchlists, rings, device, ingest, summaries, investigation, maintenance, system, agents, outreach. WebSocket `GET /ws/risk_signals`: add/remove subscriber, push new risk_signal payloads. Routes: `GET /` (links to docs, health, demo), `GET /health` → `{"status":"ok"}`. Lifespan: no-op.

- **`api/config.py`**  
  Pydantic Settings: `supabase_url`, `supabase_service_role_key`, `database_url`, `jwt_secret`, `neo4j_uri`, `neo4j_user`, `neo4j_password`. Loads from `.env`; `extra = "ignore"`.

- **`api/deps.py`**  
  Dependencies: `get_supabase()` — creates Supabase client from config, raises 503 if URL/key missing. `get_current_user_id(credentials, supabase)` — validates JWT via `supabase.auth.get_user`, returns user id or None. `require_user(user_id)` — raises 401 if None. Uses `HTTPBearer` and `OAuth2PasswordBearer` with `auto_error=False`.

**State and pipeline**

- **`api/graph_state.py`**  
  `AnchorState`: Pydantic model for LangGraph state (household_id, time_range_*, ingested_events, session_ids, normalized, utterances, entities, mentions, relationships, graph_updated, risk_scores, explanations, consent_*, watchlists, escalation_draft, persisted, needs_review, severity_threshold, consent_state, time_to_flag, logs). `append_log(state, msg)` appends to `state["logs"]`.

- **`api/pipeline.py`**  
  LangGraph pipeline: `ingest` → `normalize` (deterministic: events sorted by session, seq) → `graph_update` → `financial_security_agent` → `risk_score` (calls **domain.risk_scoring_service.score_risk**; explicit rule-only fallback when `model_available=false`) → `explain` (motifs + model_subgraph with model_evidence_quality) → `consent_gate` → `watchlist` → `escalation_draft` → `persist` (in-memory only; worker or Financial Agent persist via **domain.risk_signal_persistence.upsert_risk_signal_compound** when run on-demand). Key: `risk_score_inference` uses shared scoring service; `generate_explanations` copies model_evidence_quality; `_embedding_centroid_watchlist` (pattern with cosine_threshold, provenance). `build_graph(checkpointer)`, `run_pipeline(...)`.

- **`api/broadcast.py`**  
  In-memory set of WebSocket subscribers. `add_subscriber(ws)`, `remove_subscriber(ws)`, `broadcast_risk_signal(payload)` — sends JSON to all subscribers (used when new risk_signal is persisted via API/agent).

**Routers**

- **`api/routers/households.py`**  
  Prefix `/households`. `POST /onboard` — create household and users row for authenticated user (idempotent if already onboarded); body `OnboardRequest(display_name?, household_name?)`. `GET /me` — return `HouseholdMe` (id, name, role, display_name) for current user; 404 if not onboarded.

- **`api/routers/graph.py`**  
  Prefix `/graph`. `GET /evidence` — build evidence subgraph from household events (GraphBuilder), return `RiskSignalDetailSubgraph` (nodes/edges). `POST /sync-neo4j` — mirror that subgraph to Neo4j (no-op if not configured). `GET /neo4j-status` — `{enabled, browser_url?, connect_url?, password?}` for UI.

- **`api/routers/sessions.py`**  
  Session list and session events (see § 10 Reference: API and UI contracts for paths).

- **`api/routers/risk_signals.py`**  
  List risk signals (phase-out open signals not updated in `max_age_days`, default 90), get by id (ETag, redaction by consent), feedback, similar incidents, deep_dive explain (see api_ui_contracts).

- **`api/routers/alerts.py`**  
  `POST /alerts/{id}/refresh` — refresh single alert (narrative, conformal, outreach draft); caregiver/admin.

- **`api/routers/protection.py`**  
  Prefix `/protection`. `GET /summary`, `GET /overview`, `GET /watchlists`, `GET /rings`, `GET /rings/{ring_id}`, `GET /reports`, `GET /reports/latest` — unified Protection page (watchlists, rings, reports).

- **`api/routers/explain.py`**  
  Prefix `/explain`. `POST ""` — body: context (pattern_members | alert_ids | top_connectors | entity_list), items [{ id, hint? }]; returns plain-English explanations for opaque IDs; uses **domain.explain_service.explain_opaque_items** (Claude when ANTHROPIC_API_KEY set).

- **`api/routers/investigation.py`**  
  Prefix `/investigation`. `POST /run` — run supervisor INGEST_PIPELINE (financial + narrative + outreach candidates); body: time_window_days?, dry_run?, use_demo_events?, enqueue?; when enqueue=true (and not dry_run/demo) inserts into processing_queue, returns job_id.

- **`api/routers/maintenance.py`**  
  Prefix `/system/maintenance`. `POST /run` — run NIGHTLY_MAINTENANCE (model_health agent); admin only. `POST /clear_risk_signals` — delete all risk signals for household; admin or caregiver.

- **`api/routers/watchlists.py`**  
  List watchlists for household.

- **`api/routers/device.py`**  
  `POST /device/sync` — device heartbeat, returns watchlist delta and upload pointers.

- **`api/routers/ingest.py`**  
  `POST /ingest/events` — validate sessions belong to household, insert events; uses domain `ingest_service.ingest_events`.

- **`api/routers/summaries.py`**  
  List weekly/session summaries.

- **`api/routers/agents.py`**  
  Prefix `/agents`. Financial: `POST /financial/run`, `GET /financial/demo`, `GET /financial/trace?run_id=`. Other agents: `POST /drift/run`, `POST /narrative/run`, `POST /ring/run`, `POST /recurring_contacts/run`, `POST /calibration/run`, `POST /redteam/run`, `POST /outreach/run` (each with dry_run; agents self-persist to agent_runs; response includes run_id, step_trace, summary_json). `GET /catalog` — registry filtered by role, consent, env. `GET /status` — last run per agent (from registry + agent_runs; last_run_summary has drift_detected, rings_found, regression_pass_rate, etc.). `GET /trace?run_id=&agent_name=` and `GET /agents/{slug}/trace?run_id=` — trace for any run.

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
  Risk signal list (phase-out open signals not updated in max_age_days; RiskSignalCard with **model_available**, **title** from explanation), detail (explanation, redaction when consent disallows; ETag/Cache-Control), submit_feedback, calibration update.

- **`domain/risk_signal_persistence.py`**  
  **risk_signal_fingerprint(signal_type, explanation)** — stable hash for dedupe. **upsert_risk_signal_compound(supabase, household_id, payload, dry_run)** — if open risk with same fingerprint exists, compound score and refresh updated_at; else insert. Used by Financial Security Agent when persisting; migration 024 adds risk_signals.fingerprint.

- **`domain/claude_risk_narrative.py`**  
  **generate_risk_signal_title_and_narrative(signal_type, severity, explanation)** — optional Claude-generated title and 2–4 sentence narrative from motifs/timeline/evidence; returns None when ANTHROPIC_API_KEY missing or call fails.

- **`domain/explain_service.py`**  
  `build_subgraph_from_explanation(explanation)` — build RiskSignalDetailSubgraph from explanation subgraph/model_subgraph. `get_similar_incidents(signal_id, household_id, supabase, top_k)` — delegates to similarity_service.

- **`domain/similarity_service.py`**  
  Similar incidents: tries **pgvector RPC** `similar_incidents_by_vector` (migration 008) when available; fallback to JSONB + Python cosine. Returns SimilarIncidentsResponse with **retrieval_provenance** (model_name, checkpoint_id, embedding_dim, timestamp) when available; `available=false, reason="model_not_run"` when no real embedding.

- **`domain/watchlist_service.py`**  
  Watchlist listing (**model_available=True** for watch_type=embedding_centroid); device sync.

- **`domain/watchlists/service.py`**  
  **list_active_watchlist_items(supabase, household_id)** — unified watchlist items from `watchlist_items` (category, type, fingerprint dedupe). **upsert_watchlist_items_batch** — normalize, upsert by fingerprint, mark superseded; used by pipeline/agents.

- **`domain/watchlists/normalize.py`**  
  **watchlist_fingerprint(category, type_, key, value_normalized)** for dedupe; **normalize_watchlist_value**.

- **`domain/graph_service.py`**  
  `normalize_events` — build utterances/entities/mentions/relationships from events (GraphBuilder); **deterministic** (events sorted by ts, seq per session).

- **`domain/risk_scoring_service.py`**  
  **Single risk scoring contract:** `score_risk(household_id, sessions, utterances, entities, mentions, relationships, devices?, events?, checkpoint_path?, explanation_score_min?)` → `RiskScoringResponse(model_available, scores, fallback_used?)`. Used by pipeline, worker, and Financial Security Agent. No silent placeholders; on failure returns `model_available=false`, empty scores. When model runs: PGExplainer inline (stable entity IDs in model_subgraph), model_evidence_quality (sparsity, edges_kept/total).

- **`domain/capability_service.py`** — Household capabilities: **get_household_capabilities**, **update_household_capabilities**. Registry: notify_sms_enabled, notify_email_enabled, device_policy_push_enabled, bank_data_connector, bank_control_capabilities (lock_card, enable_alerts, etc.). Reads/writes `household_capabilities` (migration 013).

- **`domain/action_dag.py`** — **build_action_graph(signal, capabilities, consent_allow_outbound)** — deterministic DAG for incident response: nodes (task_type: verify_with_elder, notify_caregiver, device_high_risk_mode_push, call_bank, etc.), edges (dependencies), capability-gated; used by playbooks/UI.

- **`domain/agents/__init__.py`**  
  Package marker.

- **`domain/agents/financial_security_agent.py`**  
  Financial Security Agent: **DEMO_EVENTS**, **get_demo_events()**; **_ingest_events**; **normalize_events** (graph_service); **_detect_risk_patterns** (calls **domain.risk_scoring_service.score_risk** for GNN path; motif + model_available; combined 0.6*rule + 0.4*model when model ran); **_watchlist_synthesis**; **run_financial_security_playbook(...)** — full playbook, persists risk_signals/watchlists/agent_runs.

- **`domain/agents/recurring_contacts_agent.py`**  
  **run_recurring_contacts_agent(household_id, supabase, dry_run?, time_window_days?)** — identifies contacts/numbers recurring across sessions or in-session; contributes watchlist candidates (recurrence weight); persist agent_runs.

- **`domain/agents/supervisor.py`**  
  **run_supervisor(household_id, supabase, run_mode, dry_run?, risk_signal_id?, ...)** — orchestrator: INGEST_PIPELINE (financial + narrative + outreach candidates), NEW_ALERT (single alert refresh), NIGHTLY_MAINTENANCE (model_health only), ADMIN_BENCH (any agent subset). Uses calibration_params, consent_state, capabilities.

- **`domain/agents/base.py`** — AgentContext, step() context manager, persist_agent_run(), upsert_risk_signal/watchlist/summary helpers. **`domain/ml_artifacts.py`** — load_checkpoint_or_none, fetch_embeddings_window, cosine_sim, centroid, cluster_embeddings, compute_mmd_or_energy_distance. **`domain/agents/registry.py`** — AGENT_SPEC (slug → tier, triggers, visibility, run_entrypoint), get_agents_catalog (filtered by role, consent, env).

- **`domain/agents/graph_drift_agent.py`**  
  **run_graph_drift_agent(...)** — multi-metric drift (centroid, MMD, PCA+KS, neighbor stability); root-cause (model_change/new_pattern/behavior_shift); opens `drift_warning` + optional summary when drift &gt; threshold; insufficient samples → report only.

- **`domain/agents/evidence_narrative_agent.py`**  
  **run_evidence_narrative_agent(..., risk_signal_ids?, risk_signal_id?, ...)** — evidence bundle + redaction; deterministic narrative + optional LLM; stores summary/narrative and narrative_evidence_only in risk_signals.explanation.

- **`domain/agents/ring_discovery_agent.py`**  
  **run_ring_discovery_agent(...)** — interaction graph (relationships + mentions); NetworkX clustering (or Neo4j when enabled); ring_candidate risk_signals; persists rings and ring_members (migration 009).

- **`domain/agents/continual_calibration_agent.py`**  
  **run_continual_calibration_agent(...)** — Platt scaling / conformal from feedback; updates household_calibration (calibration_params, last_calibrated_at — migration 010); ECE report.

- **`domain/agents/synthetic_redteam_agent.py`**  
  **run_synthetic_redteam_agent(...)** — scenario DSL (themes + variants); sandbox pipeline; regression (similar incidents, evidence subgraph); summary: scenarios_generated, regression_pass_rate, failing_cases.

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
- **`src/app/(dashboard)/alerts/page.tsx`**, **alerts/[id]/page.tsx`**, **alert-detail-content.tsx** — Risk signals list and detail (timeline, graph, similar incidents, feedback). **Frontend:** Evidence-only badge when explanation.narrative_evidence_only; View ring / Drift warning badges for signal_type ring_candidate and drift_warning.
- **`src/app/(dashboard)/sessions/page.tsx`**, **sessions/[id]/** — Sessions and events.
- **`src/app/(dashboard)/watchlists/page.tsx`** — Watchlists.
- **`src/app/(dashboard)/summaries/page.tsx`** — Weekly summaries.
- **`src/app/(dashboard)/graph/page.tsx`** — Graph view (evidence subgraph, Sync to Neo4j, Open in Neo4j Browser).
- **`src/app/(dashboard)/ingest/page.tsx`** — Event ingest.
- **`src/app/(dashboard)/agents/page.tsx`** — Agent center (dry run, trace). **Frontend:** Renders last_run_summary per agent (drift_detected/metrics, rings_found, regression_pass_rate, failing_cases); Run / Dry run per agent; View trace by run_id.
- **`src/app/(dashboard)/elder/page.tsx`** — Elder view.
- **`src/app/(dashboard)/replay/page.tsx`** — Scenario replay (score chart, graph, trace).
- **`src/lib/api/client.ts`**, **schemas.ts** — API client and types.
- **`src/components/dashboard-nav.tsx`**, **graph-evidence.tsx** — Nav and graph viz.
- **`public/fixtures/`** — Demo mode JSON (risk_signals, sessions, watchlists, etc.); optional samples in `fixtures/archive/`.

See **§ 12 Reference: Web app** below for stack, env, routes, and demo mode.

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

- **`db/migrations/001_initial_schema.sql`** … **014_narrative_reports.sql`**  
  Incremental migrations. **007**: extended risk_signal_embeddings. **008**: optional pgvector (similar_incidents_by_vector). **009**: rings, ring_members (Ring Discovery). **010**: household_calibration.calibration_params, last_calibrated_at (Continual Calibration). **011**: user_can_contact() for outreach RLS. **012**: outbound_actions, caregiver_contacts. **013**: action_playbooks_capabilities_incident (household_capabilities, action_playbooks, incident_packets); outbound_contact_safe_display. **014**: narrative_reports (Evidence Narrative “View report”).

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
- **`run_agent_cron.py`** — Scheduled agents: `--household-id`, `--agent drift|narrative|ring|calibration|redteam`, optional `--no-dry-run`; uses ANCHOR_AGENT_CRON_DRY_RUN (default true).

**Archives (optional / reference only):** `scripts/archive/` (e.g. stress_supervisor_matrix.py), `db/archive/` (one-off SQL samples), `apps/web/public/fixtures/archive/` (scenario_replay, graph_evidence samples). See each archive README.

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

See **§ 13 Reference: Tests** below for full layout and spec/strict test descriptions.

### 2.8 Root docs

- **SETUP.md** — Single setup: Supabase, env, auth, run API/web/worker, Neo4j, Notify caregiver & Action plan, verification, troubleshooting.
- **README_EXTENDED.md** — This file; includes reference sections 7–11 (tech/training, schema, event packet, API contracts, agents).
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
| **README.md** | Quick start, layout, run steps, testing, Makefile, tech stack. |
| **README_EXTENDED.md** | This file — full codebase reference + schema, event packet, API contracts, training, agents, web app (§ 12), tests (§ 13). |
| **SETUP.md** | Single setup: Supabase, env, auth, run API/web/worker, Neo4j, Notify caregiver & Action plan, verification, troubleshooting. |

---

## 7. Reference: Tech stack, training, queue, seed, connectors

**Tech stack:** Backend: Python 3.11+, FastAPI, Pydantic, Supabase, LangGraph. ML: PyTorch 2.2+, PyG (HGT), Modal. Frontend: Next.js 14, React, Tailwind, shadcn/Radix, Zustand, TanStack Query, Recharts, React Flow. Optional: Neo4j, pgvector, OpenAI/Anthropic, Twilio/SendGrid/SMTP, Plaid.

**Training:** Train HGT on foreign data (synthetic, benchmark, or exported Supabase); run inference with `ANCHOR_ML_CHECKPOINT_PATH`. Local: `python -m ml.train --config ml/configs/hgt_baseline.yaml --data-dir data/synthetic`. Structured synthetic: `--dataset structured_synthetic --run-dir runs/hgt_structured_01`. Export from Supabase: `python scripts/export_supabase_for_gnn.py --household-id <uuid> --output data/supabase_export`. Modal: `modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml`; Elliptic: `make modal-train-elliptic`.

**Queue:** Heavy work (investigation) can run off-API via `processing_queue` (migrations 018, 019). Enqueue: `POST /investigation/run` with `enqueue=true`. Process: Modal `modal deploy modal_queue.py` (secret `anchor-supabase`) or local `python -m worker.main --poll --poll-interval 30`.

**Seed data:** `PYTHONPATH=apps/api:. python scripts/seed_supabase_data.py` (set `config/demo_placeholder.json` or `SEED_USER_ID`). Options: `--dry-run`, `--output-json`, `--sessions`, `--days-back`. After seeding: `python -m worker.main --household-id <id> --once`. **config/demo_placeholder.json:** Set `user_id` and `household_id` to your Supabase Auth user and household; used by seed_supabase_data.py, load_elderly_conversations.py --demo, run_financial_agent_demo.py, demo_replay.py. Env override: `DEMO_USER_ID`, `DEMO_HOUSEHOLD_ID`.

**Connectors:** Plaid/open banking mostly read-only; control actions gated by `household_capabilities.bank_control_capabilities`. Env: `PLAID_CLIENT_ID`, `PLAID_SECRET` if using connector endpoints.

**Demo:** Scenario replay uses time encoding, k-hop subgraph, similar incidents, motifs + PGExplainer. Demo mode: toggle uses `apps/web/public/fixtures/`; no API. `GET /agents/financial/demo` for backend replay.

**Run artifacts (runs/):** After training, each run dir (e.g. runs/hgt_synth_01/) has config.yaml, cmd.txt, history.jsonl, metrics.json, best.pt, preds_*.npz, embeddings_*.npz, calibration.json, explanations/, motifs.json, graph_stats.json. **whitepaper/** holds plot-ready CSVs (loss.csv, pr_curve_test.csv, roc_curve_test.csv, motifs.csv, explanations_edges.csv) for LaTeX pgfplots or Python/matplotlib.

---

## 8. Reference: Schema (Supabase)

**Core:** households, users (id = auth.uid(), household_id, role, display_name), devices, sessions (consent_state jsonb), events (session_id, device_id, ts, seq, event_type, payload, text_redacted).

**Derived:** utterances, summaries, entities, mentions, relationships, risk_signals (fingerprint for compound upsert — migration 024), watchlists, watchlist_items (020), processing_queue (018–019), device_sync_state, feedback, agent_runs (step_trace), risk_signal_embeddings (dim, model_name, has_embedding), household_calibration, narrative_reports, outbound_actions, caregiver_contacts, rings (fingerprint 021).

**Consent:** sessions.consent_state; helper `user_can_contact()` (011). **RLS:** Users see only rows where household_id matches their users.household_id.

---

## 9. Reference: Event packet (edge → Supabase)

**Per-event:** session_id, device_id, ts, seq, event_type, payload_version, payload (jsonb). **Types:** final_asr, intent, device_state, transaction_detected, payee_added, bank_alert_received, etc. **Batch:** `POST /ingest/events` body `{ "events": [ ... ] }`; response ingested count, session_ids, rejected. **Contract:** Pydantic in `api.schemas` (EventPacket). No graph mutation if ASR/intent confidence below threshold (`config.graph_policy`).

---

## 10. Reference: API and UI contracts

**Auth:** Supabase Auth; JWT in `Authorization: Bearer <token>`. After sign-in: `GET /households/me` → household_id, role, display_name.

**Key endpoints:** POST /households/onboard; GET /sessions?from=&to=, GET /sessions/{id}/events; GET/POST /risk_signals (list, detail, feedback, similar); GET /watchlists; POST /device/sync, POST /ingest/events; GET /summaries; POST /agents/financial/run, GET /agents/financial/demo, POST /agents/{drift,narrative,ring,calibration,redteam,outreach}/run; GET /agents/catalog, GET /agents/status, GET /agents/trace; POST /actions/outreach, GET /actions/outreach, GET /actions/outreach/summary; POST /explain; POST /investigation/run (enqueue?); POST /alerts/{id}/refresh; POST /system/maintenance/run; GET /protection/* (summary, watchlists, rings, reports).

**WebSocket:** `WS /ws/risk_signals` — message `{ type: "risk_signal", id, household_id, ts, signal_type, severity, score }`.

**Risk signal detail:** explanation (summary, motif_tags, timeline_snippet, model_available, model_subgraph?), recommended_action (checklist); subgraph nodes/edges for viz. Similar incidents: `available`, `reason` (e.g. "model_not_run"), `similar[]`. **UI data:** HouseholdMe, session list, events (text_redacted), risk card/detail (Evidence-only badge when narrative_evidence_only; View ring for ring_candidate; Drift warning for drift_warning).

---

## 11. Reference: Agents (product)

**Supervisor** (`domain/agents/supervisor`): INGEST_PIPELINE (Run Investigation: financial + narrative + outreach candidates), NEW_ALERT (single alert refresh), NIGHTLY_MAINTENANCE (model_health: drift + calibration), ADMIN_BENCH (any agent subset). **Endpoints:** POST /investigation/run, POST /alerts/{id}/refresh, POST /system/maintenance/run, GET /agents/catalog.

**Agents:** Financial Security (POST /agents/financial/run, demo, trace), Evidence Narrative (hidden in catalog), Ring Discovery, Caregiver Outreach (POST /actions/outreach, POST /agents/outreach/run), Model Health (maintenance/run), Graph Drift, Continual Calibration, Synthetic Red-Team, Recurring Contacts. All: dry_run, step_trace and summary_json in agent_runs; GET /agents/status, GET /agents/trace.

---

## 12. Reference: Web app (apps/web)

Next.js 14 (App Router) frontend; integrates with FastAPI + Supabase. **Stack:** Next.js 14, TypeScript, TailwindCSS, shadcn/ui, Zustand, TanStack Query, React Flow, Recharts, Framer Motion, Supabase Auth, WebSocket `/ws/risk_signals`.

**Env (`apps/web/.env.local`):** `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` (required for non-demo). Optional: `NEXT_PUBLIC_DEMO_MODE`, `NEXT_PUBLIC_APP_URL`. See SETUP.md § 2.3.

**Routes:** `/` landing; `/login`, `/signup`, `/onboard`, `/logout`; `/dashboard` caregiver home; `/alerts` list, `/alerts/[id]` investigation (timeline, graph, similar incidents, feedback, Evidence-only / View ring / Drift warning badges); `/sessions`, `/sessions/[id]`; `/watchlists`, `/summaries`, `/ingest`; `/graph` evidence + Sync to Neo4j; `/agents` Agent Center (dry run, trace, last-run summary per agent); `/elder` elder view; `/replay` Scenario Replay.

**Demo mode:** `NEXT_PUBLIC_DEMO_MODE=true` or sidebar toggle. Data from `apps/web/public/fixtures/`: household_me.json, risk_signals.json, risk_signal_detail.json, sessions.json, session_events.json, watchlists.json, summaries.json. Optional samples in `fixtures/archive/` (scenario_replay, graph_evidence).

**Frontend–backend:** Agents page uses `GET /agents/status` (last_run_summary: drift_detected, rings_found, regression_pass_rate); alert detail uses explanation.narrative_evidence_only, signal_type ring_candidate/drift_warning. **Roles:** Elder = minimal summary, share toggle; Caregiver/Admin = full dashboard, alerts, graph, agents.

---

## 13. Reference: Tests

Run: `pip install -e ".[ml]"` then `make test`, or `ruff check apps ml tests && pytest tests apps/api/tests apps/worker/tests ml/tests -v`. **260+ tests** (config, ML, API, pipeline, worker, scripts, Modal, routers, spec, strict). Discovery: `tests/` plus `apps/api/tests`, `apps/worker/tests`, `ml/tests` (pyproject.toml).

**Layout (summary):** Config (settings, graph); ML (config, time_encoding, subgraph, builder, models, train, inference, cache, continual, motifs, explainers, train_elliptic, pg_explainer); API (main, deps, config, state, schemas, broadcast); Pipeline (e2e, nodes, build_graph, financial node); Financial agent; Routers (agents, risk_signals, device, ingest, sessions, etc.); Modal; Worker; Broadcast; Spec tests (pipeline_spec, ml_train_spec, graph_builder_spec) — documented behavior, exact formulas/thresholds. **Strict tests** (`test_implementation_strict.py`): exact behavior and failure paths; fix code, not the test. GNN e2e: pipeline with checkpoint (optional `--train`). Tests that need torch/PyG use `pytest.importorskip(...)`.

---

*Anchor — Voice companion backend with Independence Graph, GNN risk scoring, and explainability.*
