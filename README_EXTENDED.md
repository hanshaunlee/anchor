# Anchor — Codebase Reference (Extended)

File-by-file and contract-level reference. Paths, endpoints, config keys, schema, and test layout are exact.

---

## 1. Overview

- **Anchor:** Backend for an edge voice companion that helps protect elders from fraud. No raw audio in repo; edge sends **structured event packets** (ASR, intents, financial events) via batch upload.
- **Flow:** `POST /ingest/events` → ingest (idempotent on `session_id, seq`) → normalize (deterministic) → Independence Graph → risk scoring (shared service) → explain → persist (compound upsert by fingerprint). **Read-only for money:** flags and recommends only.

---

## 2. Repository structure (exact paths)

### 2.1 Root

| File | Purpose |
|------|---------|
| `README.md` | Concise overview, diagram, quick start, links. |
| `README_EXTENDED.md` | This file. |
| `pyproject.toml` | Package `anchor`, Python ≥3.11; deps: fastapi, uvicorn, supabase, pydantic, langgraph, torch, torch-geometric, modal, pytest, ruff. Optional `[ml]`: torch-scatter, torch-sparse, torch-cluster. Optional `[db]`: psycopg2-binary. `packages`: apps*, ml*, db*. Ruff line-length 100, py311. Pytest: asyncio_mode=auto, testpaths = tests, apps/api/tests, apps/worker/tests, ml/tests. |
| `Makefile` | install, test, lint, migrate, dev-api, dev-worker, synth, train, modal-train, modal-train-elliptic. |
| `modal_app.py` | Modal app entrypoint; ML training uses `ml/modal_train.py`, `ml/modal_train_elliptic.py`. |
| `.env` | SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DATABASE_URL, JWT_SECRET; optional NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ANTHROPIC_API_KEY, ANCHOR_ML_CHECKPOINT_PATH, ANCHOR_* pipeline/agent overrides. |

### 2.2 `apps/api/` (API lives under `apps/api/api/` and `apps/api/domain/`)

**Entrypoint and config**

- **`apps/api/api/main.py`**  
  FastAPI app "Anchor API", CORS enabled. Loads `.env` from repo root (Path `__file__` → parents[3]). Routers included (order): households, graph, sessions, risk_signals, alerts, capabilities, playbooks, incident_packets, connectors, protection, explain, watchlists, rings, device, ingest, summaries, investigation, maintenance, system, agents, outreach. WebSocket: `GET /ws/risk_signals` — add/remove subscriber, push risk_signal payloads. Routes: `GET /` (links to docs, health, demo_no_auth), `GET /health` → `{"status":"ok"}`. Lifespan: no-op.

- **`apps/api/api/config.py`**  
  Pydantic Settings: `supabase_url`, `supabase_service_role_key`, `database_url`, `jwt_secret`, `neo4j_uri`, `neo4j_user`, `neo4j_password`. Loads from `.env`; `extra = "ignore"`.

- **`apps/api/api/deps.py`**  
  `get_supabase()` (503 if URL/key missing), `get_current_user_id(credentials, supabase)` (JWT via supabase.auth.get_user), `require_user(user_id)` (401 if None). HTTPBearer and OAuth2PasswordBearer with `auto_error=False`.

**State and pipeline**

- **`apps/api/api/graph_state.py`**  
  `AnchorState`: household_id, time_range_*, ingested_events, session_ids, normalized, utterances, entities, mentions, relationships, graph_updated, risk_scores, explanations, consent_*, watchlists, escalation_draft, persisted, needs_review, severity_threshold, consent_state, time_to_flag, logs. `append_log(state, msg)`.

- **`apps/api/api/pipeline.py`**  
  LangGraph: ingest → normalize → graph_update → financial_security_agent → risk_score (calls `domain.risk_scoring_service.score_risk`) → explain → consent_gate → watchlist → escalation_draft → persist. `build_graph(checkpointer)`, `run_pipeline(...)`.

- **`apps/api/api/broadcast.py`**  
  In-memory WebSocket subscribers: `add_subscriber(ws)`, `remove_subscriber(ws)`, `broadcast_risk_signal(payload)`.

**Routers (prefix and paths)**

| Router file | Prefix | Endpoints (method + path) |
|-------------|--------|---------------------------|
| `routers/households.py` | `/households` | POST `/onboard`, GET `/me`, GET `/me/consent`, PATCH `/me/consent`, GET `/me/contacts`, POST `/me/contacts` |
| `routers/graph.py` | `/graph` | GET `/evidence`, POST `/sync-neo4j`, GET `/neo4j-status` |
| `routers/sessions.py` | `/sessions` | GET `` (list), GET `/{session_id}/events` |
| `routers/risk_signals.py` | `/risk_signals` | GET `` (list), GET `/{signal_id}`, GET `/{signal_id}/page`, GET `/{signal_id}/similar`, POST `/{signal_id}/explain/deep_dive`, POST `/{signal_id}/refresh`, POST `/{signal_id}/feedback`, GET `/{signal_id}/playbook` |
| `routers/alerts.py` | `/alerts` | POST `/{id}/refresh` |
| `routers/capabilities.py` | `/capabilities` | GET `/me`, PATCH `` |
| `routers/playbooks.py` | `/playbooks` | GET `/{playbook_id}`, POST `/{playbook_id}/tasks/{task_id}/complete` |
| `routers/incident_packets.py` | `/incident_packets` | GET `/{packet_id}` |
| `routers/connectors.py` | `/connectors` | GET `/plaid/link_token`, POST `/plaid/exchange_public_token`, POST `/plaid/sync_transactions` |
| `routers/protection.py` | `/protection` | GET `/summary`, GET `/overview`, GET `/watchlists`, GET `/rings`, GET `/rings/{ring_id}`, GET `/reports`, GET `/reports/latest` |
| `routers/explain.py` | `/explain` | POST `` |
| `routers/watchlists.py` | `/watchlists` | GET `` |
| `routers/rings.py` | `/rings` | GET ``, GET `/{ring_id}` |
| `routers/device.py` | `/device` | POST `/sync` |
| `routers/ingest.py` | `/ingest` | POST `/events` |
| `routers/summaries.py` | `/summaries` | GET `` |
| `routers/investigation.py` | `/investigation` | POST `/run` |
| `routers/maintenance.py` | `/system/maintenance` | POST `/clear_risk_signals`, POST `/run` |
| `routers/system.py` | `/system` | POST `/run_ingest_pipeline` |
| `routers/agents.py` | `/agents` | POST `/financial/run`, GET `/financial/demo`, GET `/financial/trace`, GET `/catalog`, GET `/status`, GET `/trace`, GET `/{agent_slug}/trace`, POST `/drift/run`, POST `/narrative/run`, GET `/narrative/report/{report_id}`, POST `/ring/run`, GET `/calibration/report`, POST `/calibration/run`, GET `/redteam/report`, POST `/redteam/run`, POST `/outreach/run`, POST `/incident-response/run` |
| `routers/outreach.py` | `/actions` | POST `/outreach/preview`, POST `/outreach/send`, POST `/outreach`, GET `/outreach`, GET `/outreach/candidates`, GET `/outreach/summary`, GET `/outreach/{action_id}` |

**Schemas**

- **`apps/api/api/schemas.py`**  
  Pydantic: event packet (EventPacket), HouseholdMe, OnboardRequest, RiskScoreItem, RiskScoringResponse, RiskSignalCard/Detail, SubgraphNode/Edge, RiskSignalDetailSubgraph, FeedbackSubmit, EmbeddingCentroidPattern/Provenance, WatchlistItem, DeviceSyncRequest/Response, SimilarIncident, RetrievalProvenance, SimilarIncidentsResponse, WeeklySummary. Contract of record for API and event packet.

**Neo4j**

- **`apps/api/api/neo4j_sync.py`**  
  `_driver()` from api.config (neo4j_uri, user, password). `sync_evidence_graph_to_neo4j(household_id, entities, relationships)` — MERGE then clear household subgraph. `neo4j_enabled()`.

**Domain (exact module paths under `apps/api/domain/`)**

- **`domain/ingest_service.py`** — `get_household_id(supabase, user_id)`, `ingest_events(...)` with upsert `on_conflict=(session_id, seq)`.
- **`domain/risk_service.py`** — List/detail risk signals (phase-out by max_age_days), feedback, calibration.
- **`domain/risk_signal_persistence.py`** — `risk_signal_fingerprint(signal_type, explanation)`, `upsert_risk_signal_compound(supabase, household_id, payload, dry_run)` (compound update or insert).
- **`domain/claude_risk_narrative.py`** — `generate_risk_signal_title_and_narrative(signal_type, severity, explanation)` when ANTHROPIC_API_KEY set.
- **`domain/explain_service.py`** — `build_subgraph_from_explanation(explanation)`, `get_similar_incidents(...)`.
- **`domain/similarity_service.py`** — pgvector RPC `similar_incidents_by_vector` (migration 008) or JSONB cosine fallback; retrieval_provenance; `available=false, reason="model_not_run"` when no embedding.
- **`domain/watchlist_service.py`** — List watchlists, device sync.
- **`domain/watchlists/service.py`** — `list_active_watchlist_items`, `upsert_watchlist_items_batch` (fingerprint dedupe).
- **`domain/watchlists/normalize.py`** — `watchlist_fingerprint(category, type_, key, value_normalized)`, `normalize_watchlist_value`.
- **`domain/graph_service.py`** — `normalize_events` (GraphBuilder; deterministic sort by session, seq).
- **`domain/risk_scoring_service.py`** — **`score_risk(household_id, *, sessions, utterances, entities, mentions, relationships, devices=None, events=None, checkpoint_path=None, explanation_score_min=None, calibration_params=None, pattern_tags=None, structural_motifs=None)`** → `RiskScoringResponse(model_available, scores[])`. Each score: raw_score, calibrated_p, rule_score, fusion_score, uncertainty, decision_rule_used, model_subgraph (entity IDs), model_evidence_quality. Uses `domain.rule_scoring.compute_rule_score` when model unavailable; `domain.explainers.pg_service.attach_pg_explanations` when model runs.
- **`domain/capability_service.py`** — get/update household_capabilities (notify_sms_enabled, bank_control_capabilities, etc.).
- **`domain/action_dag.py`** — `build_action_graph(signal, capabilities, consent_allow_outbound)` — DAG for verify_with_elder, notify_caregiver, device_high_risk_mode_push, etc.
- **`domain/agents/base.py`** — AgentContext, step(), persist_agent_run(), upsert helpers. **`domain/ml_artifacts.py`** — load_checkpoint_or_none, centroid, cluster_embeddings, compute_mmd_or_energy_distance.
- **`domain/agents/registry.py`** — AGENT_SPEC (slug → tier, triggers, visibility, run_entrypoint), get_agents_catalog (filtered by role, consent, env).
- **`domain/agents/supervisor.py`** — `run_supervisor(household_id, supabase, run_mode, ...)`: INGEST_PIPELINE, NEW_ALERT, NIGHTLY_MAINTENANCE, ADMIN_BENCH.
- **`domain/agents/financial_security_agent.py`** — run_financial_security_playbook: ingest → normalize → _detect_risk_patterns (risk_scoring_service) → _watchlist_synthesis → persist.
- **`domain/agents/graph_drift_agent.py`** — run_graph_drift_agent: centroid/MMD/PCA+KS/neighbor stability; drift_warning risk_signal; conformal invalidation.
- **`domain/agents/evidence_narrative_agent.py`** — run_evidence_narrative_agent; narrative_evidence_only in explanation.
- **`domain/agents/ring_discovery_agent.py`** — run_ring_discovery_agent; rings, ring_members (migration 009).
- **`domain/agents/continual_calibration_agent.py`** — Platt + conformal; household_calibration (migration 010).
- **`domain/agents/synthetic_redteam_agent.py`** — scenario DSL; sandbox pipeline; regression pass_rate, failing_cases.
- **`domain/agents/recurring_contacts_agent.py`** — run_recurring_contacts_agent; watchlist candidates by recurrence.

### 2.3 `apps/worker/`

- **`apps/worker/main.py`** — Entrypoint `python -m worker.main`; `--household-id`, `--once`; calls `worker.jobs.run_pipeline(supabase, household_id)` when once+household-id.
- **`apps/worker/worker/jobs.py`** — ingest_events_batch, run_graph_builder, run_risk_inference (domain.risk_scoring_service.score_risk), run_pipeline; persist risk_signals, risk_signal_embeddings, watchlists, agent_runs.step_trace; optional Neo4j sync.

### 2.4 `apps/web/`

Next.js 14 App Router. Key paths:

- **`apps/web/src/app/page.tsx`** — Landing.
- **`apps/web/src/app/(auth)/login/page.tsx`**, `signup/`, `onboard/`, `logout/` — Auth.
- **`apps/web/src/app/(dashboard)/dashboard/page.tsx`** — Caregiver home.
- **`apps/web/src/app/(dashboard)/alerts/page.tsx`**, `alerts/[id]/page.tsx`**, **`alert-detail-content.tsx`** — Alerts list/detail (timeline, graph, similar, feedback; Evidence-only, View ring, Drift warning badges).
- **`apps/web/src/app/(dashboard)/sessions/`**, **watchlists/**, **summaries/**, **graph/**, **ingest/**, **agents/**, **elder/**, **replay/`** — As named.
- **`apps/web/src/lib/api/client.ts`**, **schemas.ts** — API client and types.
- **`apps/web/public/fixtures/`** — Demo mode JSON (household_me.json, risk_signals.json, sessions.json, etc.).

### 2.5 `ml/`

- **`ml/train.py`** — CLI: HGT (or GPS/FraudGT) on synthetic/HGB; focal_loss; checkpoint: in_channels, metadata, model_state, hidden_channels, out_channels, num_layers, heads, target_node_type.
- **`ml/inference.py`** — `load_model(checkpoint_path, device)`, `run_inference(..., return_embeddings)`; optional GNNExplainer.
- **`ml/train_elliptic.py`** — Elliptic dataset; FraudGT-style.
- **`ml/config.py`** — get_train_config() from YAML (e.g. configs/hgt_baseline.yaml).
- **`ml/models/hgt_baseline.py`** — HGTBaseline; forward_hetero_data_with_hidden for embeddings/explainers.
- **`ml/models/gps_model.py`** — GraphGPS (not default in pipeline).
- **`ml/models/fraud_gt_style.py`** — Elliptic pipeline only.
- **`ml/graph/builder.py`** — GraphBuilder: process_events (final_asr, intent, transaction_detected, payee_added, bank_alert_received, watchlist_hit, slot_to_entity); get_utterance_list, get_entity_list, get_mention_list, get_relationship_list; build_hetero_from_tables (same schema as training).
- **`ml/graph/subgraph.py`** — k-hop extract, time_window.
- **`ml/graph/time_encoding.py`** — Sinusoidal time encoding.
- **`ml/explainers/motifs.py`** — extract_motifs (urgency_topics, sensitive_intents from config.graph).
- **`ml/explainers/gnn_explainer.py`**, **`ml/explainers/pg_explainer.py`** — GNNExplainer-style, PGExplainerStyle (top_edges, minimal_subgraph_node_ids).
- **`ml/continual/finetune_last_layer.py`** — finetune_last_layer_stub.
- **`ml/cache/embeddings.py`** — EmbeddingCache, get_similar_sessions.
- **`ml/modal_train.py`**, **`ml/modal_train_elliptic.py`** — Modal training entrypoints.

### 2.6 `config/`

- **`config/settings.py`**  
  **PipelineSettings** (env_prefix `ANCHOR_`): risk_score_threshold=0.5, explanation_score_min=0.4, watchlist_score_min=0.5, escalation_score_min=0.6, persist_score_min=0.2, min_severity_to_persist=1, severity_threshold=4, timeline_snippet_max=6, consent_share_key, consent_watchlist_key, consent_allow_outbound_key, default_consent_share=True, default_consent_watchlist=True, default_consent_allow_outbound=False, asr_confidence_min_for_graph=0.0.  
  **MLSettings** (env_prefix `ANCHOR_ML_`): embedding_dim=128, risk_inference_entity_cap=100, calibration_adjust_step=0.1, calibration_adjust_cap=2.0, calibration_adjust_floor=-0.5, calibration_true_positive_step=-0.05, model_version_tag="v0", checkpoint_path="runs/hgt_baseline/best.pt", explainer_epochs=50.  
  **AgentSettings** (env_prefix `ANCHOR_AGENT_`): evidence_signal_limit=20, evidence_llm_max_tokens=400; drift_window_recent_days=3, drift_window_baseline_days=14, drift_threshold=0.15, drift_min_samples_per_window=10, drift_mmd_threshold=0.2, drift_ks_threshold=0.3; calibration_min_labeled=10, calibration_target_fpr=0.1, calibration_ece_bins=10; ring_min_community_size=2, ring_top_rings=10; similar_incidents_window_days=90, similar_incidents_top_k=5; risk_scoring_top_k_edges=20.  
  **NotifySettings** (ANCHOR_NOTIFY_): provider="mock" \| twilio \| sendgrid \| smtp.  
  **WorkerSettings** (ANCHOR_WORKER_): outreach_auto_trigger=True.  
  Accessors: get_pipeline_settings(), get_ml_settings(), get_agent_settings(), get_notify_settings(), get_worker_settings() (all lru_cache). get_ml_config_path() → env ANCHOR_ML_CONFIG or "configs/hgt_baseline.yaml".

- **`config/graph.py`** — get_graph_config() (lru_cache): node_types, edge_types, entity_type_map, slot_to_entity, event_types, speaker_roles, base_feature_dims, time_encoding_dim, edge_attr_dim, co_occurrence_window_sec, person_ids, urgency_topics, sensitive_intents. Override via YAML (ANCHOR_GRAPH_CONFIG).

- **`config/graph_policy.py`** — allow_graph_mutation_for_event(event, confidence_min): final_asr and intent gated by payload.confidence >= confidence_min (PipelineSettings.asr_confidence_min_for_graph); other event types allowed.

### 2.7 `db/`

- **`db/bootstrap_supabase.sql`** — Full schema (run once): enums user_role, session_mode, speaker_type, entity_type_enum, risk_signal_status, feedback_label; tables households, users, devices, sessions, events (UNIQUE session_id, seq), utterances, summaries, entities, mentions, relationships, risk_signals, watchlists, device_sync_state, feedback, session_embeddings, risk_signal_embeddings, household_calibration, agent_runs; indexes; RLS; helpers user_household_id, user_role.

**Migrations (exact filenames, order 001–024):**

| # | Filename | Purpose |
|---|----------|---------|
| 001 | 001_initial_schema.sql | Core schema (already in bootstrap) |
| 002 | 002_rls.sql | RLS policies |
| 003 | 003_risk_signal_embeddings.sql | risk_signal_embeddings |
| 004 | 004_rls_embeddings_calibration.sql | RLS for embeddings/calibration |
| 005 | 005_agent_runs.sql | agent_runs |
| 006 | 006_agent_runs_step_trace.sql | step_trace on agent_runs |
| 007 | 007_risk_signal_embeddings_extended.sql | Extended embeddings columns |
| 008 | 008_pgvector_embeddings.sql | pgvector, similar_incidents_by_vector RPC |
| 009 | 009_rings.sql | rings, ring_members |
| 010 | 010_household_calibration_params.sql | household_calibration (platt, conformal) |
| 011 | 011_role_consent_helpers.sql | user_can_contact() |
| 012 | 012_outbound_actions_caregiver_contacts.sql | outbound_actions, caregiver_contacts |
| 013a | 013_action_playbooks_capabilities_incident.sql | household_capabilities, action_playbooks, incident_packets |
| 013b | 013_outbound_contact_safe_display.sql | outbound_contact_safe_display |
| 014 | 014_narrative_reports.sql | narrative_reports |
| 015 | 015_outbound_actions_conformal_auto_send.sql | Conformal auto-send |
| 016 | 016_rpc_alert_page_and_investigation_context.sql | RPCs for alert page / investigation |
| 017 | 017_performance_indexes.sql | Performance indexes |
| 018 | 018_processing_queue.sql | processing_queue |
| 019 | 019_processing_queue_dedupe_retry.sql | Dedupe/retry for queue |
| 020 | 020_watchlist_items.sql | watchlist_items |
| 021 | 021_rings_fingerprint_canonical.sql | rings fingerprint/canonical |
| 022 | 022_embedding_vector_128.sql | Embedding vector 128 dim |
| 023 | 023_protection_rings_watchlist_columns.sql | Protection/rings/watchlist columns |
| 024 | 024_risk_signals_fingerprint.sql | risk_signals.fingerprint (compound upsert) |

### 2.8 `scripts/`

- **run_api.sh** — PYTHONPATH=apps/api; uvicorn api.main:app (reload on apps/api, config).
- **run_worker.sh** — PYTHONPATH=.:apps:apps/api; python -m worker.main.
- **start_neo4j.sh** — Docker Neo4j (NEO4J_IMAGE, NEO4J_CONTAINER, NEO4J_PASSWORD).
- **synthetic_scenarios.py** — Generate scenario events; --household-id, --output.
- **demo_replay.py** — Pipeline on demo events; --ui --launch-ui.
- **run_financial_agent_demo.py** — Financial playbook on demo events (no API).
- **seed_supabase_data.py** — Full seed; --dry-run, --output-json, --household-id, SEED_USER_ID; config/demo_placeholder.json (user_id, household_id).
- **run_gnn_e2e.py** — Train + pipeline; --skip-train, --train.
- **run_migration.py** — DATABASE_URL.
- **run_replay_time_to_flag.py** — time_to_flag metrics.
- **run_agent_cron.py** — --household-id, --agent drift|narrative|ring|calibration|redteam; ANCHOR_AGENT_CRON_DRY_RUN (default true).

### 2.9 `tests/` (exact test file names)

| File | Scope |
|------|--------|
| test_api_main.py, test_api_deps.py, test_api_config.py | API app, deps, config |
| test_broadcast.py, test_broadcast_multi.py | WebSocket subscribers |
| test_config_init.py, test_config_settings.py, test_config_settings_extended.py, test_config_graph.py | Config, settings, graph |
| test_graph_state.py, test_schemas.py, test_schemas_extended.py | State, Pydantic schemas |
| test_pipeline.py, test_pipeline_nodes.py, test_pipeline_build_graph.py, test_pipeline_financial_node.py, test_pipeline_spec.py | Pipeline e2e, nodes, spec |
| test_financial_agent.py, test_financial_agent_embeddings.py | Financial agent, embeddings |
| test_domain_agents.py, test_supervisor.py, test_agents_catalog.py, test_model_health_agent.py, test_incident_response_agent.py | Domain agents, supervisor, catalog |
| test_risk_service.py, test_risk_scoring_service.py, test_similarity_service.py, test_explain_service.py | Risk, similarity, explain |
| test_graph_service.py, test_graph_policy.py, test_graph_builder.py, test_graph_builder_units.py, test_graph_builder_spec.py | Graph service, policy, builder |
| test_action_dag.py, test_playbooks_capabilities_device.py, test_protection_rings_watchlist.py | Action DAG, playbooks, protection |
| test_routers_ingest.py, test_routers_device.py, test_routers_sessions_events.py, test_routers_agents.py, test_routers_agents_extended.py, test_routers_risk_signals_detail.py, test_routers_extra.py, test_api_routers.py | Routers |
| test_worker_main.py, test_worker_jobs.py, test_worker_jobs_extended.py | Worker |
| test_ml_config.py, test_ml_time_encoding.py, test_ml_subgraph.py, test_ml_models.py, test_ml_train.py, test_ml_train_spec.py, test_ml_train_elliptic.py, test_ml_inference.py, test_ml_motifs.py, test_ml_explainers_gnn.py, test_ml_explainers_pg.py, test_ml_fraud_gt_style.py, test_ml_continual.py, test_ml_cache_embeddings.py | ML config, graph, models, train, inference, explainers, continual, cache |
| test_implementation_strict.py | Strict behavior (ingest, deps 503/401, financial node exception, etc.) — fix code not test |
| test_pipeline_spec.py, test_ml_train_spec.py, test_graph_builder_spec.py | Spec tests (documented formulas/thresholds) |
| test_gnn_e2e.py, test_gnn_product_loop.py | GNN e2e, product loop |
| test_contract_config.py, test_contract_api_routes.py, test_contract_domain_public.py, test_contract_ml_public.py | Contract tests |
| test_flow_*.py | Flow tests (ingest_risk_ui, bulk_contract, bulk_parametrized, worker_and_agents, gnn_consumers, pipeline_state) |
| test_conformal_escalation.py, test_conformal_rule_independence.py | Conformal escalation, independence |
| test_structural_motifs_and_independence.py | Structural motifs, independence |
| test_outreach.py, test_synthetic_scenarios.py, test_scripts_demo.py, test_modal.py, test_cohesion.py, test_time_utils.py | Outreach, synthetic, demo script, Modal, cohesion, time |

Run: `make test` or `ruff check apps ml tests && pytest tests apps/api/tests apps/worker/tests ml/tests -v`. Discovery per pyproject.toml testpaths.

---

## 3. Data flow

1. Edge → **POST /ingest/events** (body `{ "events": [ { session_id, device_id, ts, seq, event_type, payload_version, payload } ] }`) → ingest_service.ingest_events (upsert on session_id, seq).
2. Worker or on-demand: jobs.run_pipeline → api.pipeline.run_pipeline → risk_score (domain.risk_scoring_service.score_risk) → persist (risk_signal_persistence.upsert_risk_signal_compound when Supabase provided).
3. API: list/detail risk_signals (phase-out max_age_days), feedback, similar (pgvector RPC or JSONB), watchlists, device/sync, agents (run, status, trace), graph/evidence, graph/sync-neo4j.
4. WebSocket **/ws/risk_signals**: broadcast_risk_signal on new persist.

---

## 4. Schema (Supabase) — core tables and key columns

- **households:** id, name, created_at
- **users:** id (auth.uid()), household_id, role (user_role), display_name, created_at, updated_at
- **devices:** id, household_id, device_type, firmware_version, last_seen_at
- **sessions:** id, household_id, device_id, started_at, ended_at, mode, consent_state (jsonb)
- **events:** id, session_id, device_id, ts, seq, event_type, payload, payload_version, text_redacted; **UNIQUE(session_id, seq)**
- **utterances:** id, session_id, ts, speaker, text, intent, confidence
- **entities:** id, household_id, entity_type (entity_type_enum), canonical, canonical_hash, meta
- **mentions:** id, session_id, utterance_id, event_id, entity_id, ts
- **relationships:** id, household_id, src_entity_id, dst_entity_id, rel_type, weight, first_seen_at, last_seen_at, evidence
- **risk_signals:** id, household_id, signal_type, severity, score, status, explanation (jsonb), fingerprint (migration 024), created_at, updated_at
- **watchlists,** **watchlist_items** (020), **rings** (009), **ring_members**, **household_calibration** (010), **agent_runs** (step_trace), **risk_signal_embeddings** (dim, model_name, has_embedding; pgvector in 008), **processing_queue** (018–019), **narrative_reports** (014), **outbound_actions,** **caregiver_contacts**

RLS: household-scoped via user_household_id(), user_role(). Consent: sessions.consent_state; user_can_contact() (011).

---

## 5. Event packet (edge → ingest)

**Per event:** session_id, device_id, ts, seq, event_type, payload_version (int), payload (jsonb). **Event types (config/graph):** final_asr, intent, device_state, transaction_detected, payee_added, bank_alert_received, watchlist_hit, etc. **Batch:** POST body `{ "events": [ ... ] }`. Response: ingested count, session_ids, rejected. Pydantic: api.schemas (EventPacket). Graph mutation: allow_graph_mutation_for_event(event, asr_confidence_min_for_graph) — final_asr and intent require payload.confidence >= threshold.

---

## 6. API and UI contracts (summary)

**Auth:** Supabase Auth; JWT `Authorization: Bearer <token>`. After sign-in: GET /households/me → HouseholdMe (id, name, role, display_name).

**Risk signal list:** Phase-out open signals not updated in max_age_days (default 90). Card: model_available, title from explanation. **Detail:** explanation (summary, motif_tags, timeline_snippet, model_subgraph?, narrative_evidence_only?), recommended_action; GET /risk_signals/{id}/similar → available, reason, similar[]; retrieval_provenance when pgvector used.

**WebSocket:** Message type `risk_signal`: id, household_id, ts, signal_type, severity, score.

**UI badges:** Evidence-only (narrative_evidence_only), View ring (ring_candidate), Drift warning (drift_warning).

---

## 7. Training, queue, seed, env

**Training:** Local: `python -m ml.train --config ml/configs/hgt_baseline.yaml --data-dir data/synthetic`. Structured: `--dataset structured_synthetic --run-dir runs/hgt_structured_01`. Modal: `modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml`. Elliptic: make modal-train-elliptic. Checkpoint path: ANCHOR_ML_CHECKPOINT_PATH or config default runs/hgt_baseline/best.pt.

**Queue:** processing_queue (018, 019). Enqueue: POST /investigation/run with enqueue=true. Process: worker --poll --poll-interval 30 or Modal modal_queue.

**Seed:** `PYTHONPATH=apps/api:. python scripts/seed_supabase_data.py`; config/demo_placeholder.json or SEED_USER_ID, DEMO_USER_ID, DEMO_HOUSEHOLD_ID.

**Env (API):** SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DATABASE_URL, JWT_SECRET; NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD; ANTHROPIC_API_KEY (optional explain/narrative). Pipeline/ML/Agent: ANCHOR_*, ANCHOR_ML_*, ANCHOR_AGENT_* (see config/settings.py). Web: NEXT_PUBLIC_API_BASE_URL, NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY, NEXT_PUBLIC_DEMO_MODE.

---

## 8. Agents (product) — entrypoints and behavior

| Agent | Entrypoint | Key behavior |
|-------|------------|--------------|
| Supervisor | run_supervisor(run_mode=INGEST_PIPELINE \| NEW_ALERT \| NIGHTLY_MAINTENANCE \| ADMIN_BENCH) | Orchestrates financial + narrative + outreach; maintenance = drift + calibration |
| Financial Security | POST /agents/financial/run, GET /agents/financial/demo, GET /agents/financial/trace | ingest → normalize → score_risk → watchlist → persist (compound upsert) |
| Graph Drift | POST /agents/drift/run | centroid/MMD/PCA+KS/neighbor stability; drift_warning; sets conformal_invalid_since |
| Evidence Narrative | POST /agents/narrative/run | narrative_evidence_only in explanation |
| Ring Discovery | POST /agents/ring/run | rings, ring_members; ring_candidate risk_signals |
| Continual Calibration | POST /agents/calibration/run | Platt + conformal; household_calibration |
| Synthetic Red-Team | POST /agents/redteam/run | Scenario DSL; regression pass_rate, failing_cases |
| Recurring Contacts | (via supervisor or cron) | Watchlist candidates by recurrence |
| Caregiver Outreach | POST /actions/outreach/preview, POST /actions/outreach/send, POST /agents/outreach/run | Consent-gated; auto when conformal + capabilities |

GET /agents/catalog (filtered by role, consent, env). GET /agents/status (last_run_summary per agent). GET /agents/trace?run_id=&agent_name= or GET /agents/{slug}/trace?run_id=.

---

## 9. Web app (apps/web)

**Stack:** Next.js 14, TypeScript, TailwindCSS, shadcn/ui, Zustand, TanStack Query, React Flow, Recharts, Framer Motion, Supabase Auth, WebSocket /ws/risk_signals.

**Routes:** /, /login, /signup, /onboard, /logout; /dashboard; /alerts, /alerts/[id]; /sessions, /sessions/[id]; /watchlists; /summaries; /ingest; /graph; /agents; /elder; /replay.

**Demo:** NEXT_PUBLIC_DEMO_MODE=true or sidebar toggle. Fixtures: apps/web/public/fixtures/*.json.

---

*Anchor — Voice companion backend with Independence Graph, GNN risk scoring, and explainability.*
