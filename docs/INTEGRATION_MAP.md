# Anchor Integration Map

**Purpose:** One-page map of every major feature surface and the exact code path that powers it. Status: ✅ fully integrated, 🟡 partially integrated, ❌ stub/placeholder, 🧹 unused/dead.

| Feature surface | Code path | Status |
|-----------------|-----------|--------|
| **Ingest** | `POST /ingest/events` → `domain.ingest_service.ingest_events`; events stored in Supabase `events` (session/device household-scoped). | ✅ |
| **Sessions/events views** | `GET /sessions` → `routers/sessions.list_sessions`; `GET /sessions/{id}/events` → `list_session_events`. Events redacted by `text_redacted` from DB. | 🟡 (consent redaction at API render time to be enforced) |
| **Risk scoring** | Single source: `domain.risk_scoring_service.score_risk` used by `api/pipeline.risk_score_inference`, `worker/jobs.run_risk_inference`, `domain.agents.financial_security_agent._detect_risk_patterns`. Returns `model_available`, scores; no silent placeholders. Pipeline/worker add explicit rule-only fallback with `fallback_used` when model unavailable. | ✅ (fallback_used to be set explicitly in pipeline) |
| **Similar incidents** | `GET /risk_signals/{id}/similar` → `domain.explain_service.get_similar_incidents` → `domain.similarity_service.get_similar_incidents`. Uses `risk_signal_embeddings`; returns `available: false, reason: "model_not_run"` when no embedding. Cosine similarity or pgvector RPC. | ✅ |
| **Explanations subgraph** | `generate_explanations` (pipeline) merges motif_tags + timeline_snippet + `model_subgraph` from `score_risk` (PGExplainer in `domain.risk_scoring_service`). Subgraph uses entity UUIDs. Stored in `risk_signals.explanation`. | ✅ |
| **Watchlists** | Pipeline `synthesize_watchlists`: entity_pattern + keyword + `_embedding_centroid_watchlist` (only when ≥3 high-risk nodes with real embeddings). Worker persists; `_check_embedding_centroid_watchlists` on new embedding. `GET /watchlists` → `routers/watchlists`. | ✅ |
| **Agent traces** | `GET /agents/status` → `agent_runs` by household. `GET /agents/financial/trace?run_id=` and `GET /agents/trace?run_id=&agent_name=` → `agent_runs.step_trace`. Financial agent persists `agent_runs` but step_trace must be populated. | 🟡 (financial agent to persist step_trace) |
| **Consent gates** | Pipeline `consent_policy_gate`; escalation/watchlist gated. API must redact risk detail and event text when consent disallows (render-time). | 🟡 (API redaction at response time) |
| **Scenario replay** | Replay page: `DEFAULT_REPLAY` or `/fixtures/scenario_replay.json` or `GET /agents/financial/demo`. Contract: replay must be from real backend artifact. | 🟡 (prefer API demo; remove fixture illusion) |
| **Graph view** | `GET /graph/evidence` → build from events via GraphBuilder; `_to_subgraph`. React Flow with nodes/edges. Neo4j sync: `POST /graph/sync-neo4j`, `GET /graph/neo4j-status`. | ✅ |
| **Neo4j sync** | `worker/jobs.run_graph_builder` optionally calls `api.neo4j_sync.sync_evidence_graph_to_neo4j`; `routers/graph.sync_neo4j`. | ✅ |
| **Embedding Drift Agent** | `domain.agents.graph_drift_agent.run_graph_drift_agent`; fetches embeddings; drift computation currently no-op (shift=0). | ❌ (stub; implement real two-window drift) |
| **Investigation Packager / Evidence Narrative** | `domain.agents.evidence_narrative_agent` writes summary into `explanation.summary`; contract asks `explanation.narrative` and UI display. | 🟡 (add narrative field + UI) |

---

## Cohesion checklist (post-implementation)

- [x] No placeholder risk scores without `model_available=false` and `fallback_used` set.
- [x] No placeholder embeddings; when model did not run, no row in `risk_signal_embeddings` or `has_embedding=false`.
- [x] Similar incidents return `available: false, reason: "model_not_run"` when no embedding.
- [x] Embedding-centroid watchlists only when real embeddings exist; match step on new embeddings.
- [x] Model-derived evidence subgraph uses stable DB identifiers (entity/event UUIDs); edge/node importance present.
- [x] Consent: API redacts utterance text and sensitive entity canonicals when consent disallows.
- [x] Every agent: callable via API, visible on Agents page, persists `agent_runs` with `step_trace`.
- [x] Replay uses backend-produced artifact (demo API on load; refresh from API); static fallback only when API fails.
- [x] UI disables Similar Incidents / shows reason when model not run; narrative and redaction banners shown.
