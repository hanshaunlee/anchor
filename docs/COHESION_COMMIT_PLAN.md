# Cohesion integration – commit-by-commit plan

Prioritized commits (max 20) to achieve a cohesive system with no orphaned agents, no fake outputs, and real GNN product loop. Integrate before deleting; delete only when no product/experimental role.

---

## PHASE 1 — Embeddings + similarity end-to-end (Financial agent)

### Commit 1 — Financial agent: persist risk_signal_embeddings when model runs
**Files:** `apps/api/domain/agents/financial_security_agent.py`
- In `_detect_risk_patterns`: capture `embedding` per node from `score_risk` response (from `response.scores[i].embedding`) and return `model_meta` (for model_name, checkpoint_id).
- Add `embedding` and optional `model_meta` to each `risk_scores` item; pass `model_meta` out of `_detect_risk_patterns`.
- In `run_financial_security_playbook` persist loop: after each `risk_signals.insert`, if that signal has a real `embedding`, upsert into `risk_signal_embeddings` with same contract as worker: `risk_signal_id`, `household_id`, `embedding`, `dim`, `model_name`, `checkpoint_id`, `has_embedding=True`, `meta`; use `_ml_settings()` for model_version_tag when model_meta missing.
- When model not available: do not write any row to `risk_signal_embeddings`; store reason in explanation meta if desired.
**Acceptance:** Unit test (see Commit 3). GET `/risk_signals/{id}/similar` returns `available=true` with results for financial-created signals when embeddings were persisted.
**Tests:** `tests/test_financial_agent_embeddings.py` (new).

### Commit 2 — Frontend: Similar Incidents schema never default available=true
**Files:** `apps/web/src/lib/api/schemas.ts`, optionally `apps/web/src/app/(dashboard)/alerts/[id]/alert-detail-content.tsx`
- In `SimilarIncidentsResponseSchema`: change `available: z.boolean().optional().default(true)` to `available: z.boolean().optional()` so backend `available: false` is never overridden.
- In alert detail: treat `similarData?.available !== true` as “unavailable” (show “Unavailable” / “Similar incidents require GNN embeddings…”).
**Acceptance:** When API returns `available: false`, UI shows “Unavailable (model not run).” and does not show a similar list. No frontend default implying similarity when backend says no.
**Tests:** Frontend type/schema test or manual verification; optional Jest test that parsed response with `available: false` yields `available === false`.

### Commit 3 — Tests: Financial agent embedding persist and similar flow
**Files:** `tests/test_financial_agent_embeddings.py` (new), `tests/conftest.py` (if needed for fixtures)
- Test: `run_financial_security_playbook` with mocked `score_risk` returning `model_available=True`, one score with `embedding=[0.1]*32` → assert one row inserted into `risk_signal_embeddings` (mock supabase) with `has_embedding=true`, `dim=32`, `model_name` set.
- Test: when `score_risk` returns `model_available=False`, no insert into `risk_signal_embeddings`.
**Acceptance:** Both tests pass. Integration: Similar incidents panel works for financial-created signals when embeddings exist.

---

## PHASE 2 — Embedding-centroid watchlists real feature

### Commit 4 — Pipeline/worker: generate centroid watchlists when enough embeddings exist
**Files:** `apps/worker/worker/jobs.py` and/or `apps/api/api/pipeline.py`, `domain/ml_artifacts.py`
- Add step (pipeline or job): aggregate embeddings (e.g. top-K high-risk), compute centroid, write `watchlists` row with `watch_type="embedding_centroid"`, `pattern`: `{ "centroid": [...], "threshold": 0.18, "metric": "cosine", "source": { "risk_signal_ids": [...], "window": "7d" } }`.
- Only create when count of real embeddings >= N (e.g. 3).
**Tests:** Unit test: centroid watchlist created only when embeddings count >= N.

### Commit 5 — Worker: trigger embedding-match alerts vs centroid watchlists
**Files:** `apps/worker/worker/jobs.py`
- In `_check_embedding_centroid_watchlists` or on new embedding: compare to centroid watchlists; if match above threshold, create `risk_signal` with `signal_type="watchlist_embedding_match"`, severity 3–4, explanation (similarity, centroid id). Persist and broadcast.
**Tests:** Unit test: match creates risk_signal. E2E script: seed embeddings → create centroid → insert new embedding → assert match signal.

### Commit 6 — UI: Watchlists page and alert detail for embedding_centroid
**Files:** `apps/web/src/...` (watchlists page, alert detail)
- Watchlists: show `embedding_centroid` with centroid provenance, threshold, last match.
- Alert detail: if `signal_type === "watchlist_embedding_match"`, show “Matched centroid watchlist” with similarity and provenance.
**Tests:** Manual or E2E.

---

## PHASE 3 — Model-derived evidence subgraph (PGExplainer + GNNExplainer)

### Commit 7 — risk_scoring_service: model_subgraph with DB-stable entity IDs
**Files:** `apps/api/domain/risk_scoring_service.py`, `ml/graph/builder.py` (if mapping needed)
- In `score_risk`, after PGExplainer: ensure `model_subgraph` uses entity UUIDs in `nodes[].id` and `edges[].src_entity_id`/`dst_entity_id` (or src/dst as UUIDs). Provide PyG index → entity UUID mapping in builder/score path.
- Pipeline `generate_explanations`: store returned `model_subgraph` in `explanation_json`.
**Tests:** Backend test: model_subgraph has >0 edges and stable IDs when model available.

### Commit 8 — GNNExplainer deep-dive: setting, endpoint, persist
**Files:** `apps/api/domain/risk_scoring_service.py` or explainer router, `ml/explainers/gnn_explainer.py`, DB migration if new column
- Add `EXPLAINER_MODE`: "pg" | "gnn" | "both". Default "pg". For severity >= 4 or on-demand, run GNNExplainer; persist `explanation.deep_dive_subgraph` (edge mask weights, node importances).
- `POST /risk_signals/{id}/explain/deep_dive?mode=gnn` → run explainer (async worker or sync with timeout), persist, return status.
**Tests:** Backend test: deep_dive endpoint persists `deep_dive_subgraph`.

### Commit 9 — UI: Graph panel model evidence + deep-dive toggle
**Files:** `apps/web/src/app/(dashboard)/alerts/[id]/alert-detail-content.tsx` (or graph component)
- Graph: rule evidence subgraph + model evidence subgraph (edge thickness = importance). Toggle PG vs GNNExplainer deep dive. If deep dive not computed: “Compute deep dive” button → call endpoint → spinner → render result.
**Tests:** UI/visual.

---

## PHASE 4 — Replay uses real backend artifacts

### Commit 10 — Replay: refresh from API uses real risk_signals + explanation subgraphs
**Files:** `apps/web/...` (replay page/component)
- “Refresh from API”: use returned `risk_signals` and their `explanation.subgraph` / `model_subgraph` to populate graph; use agent `step_trace` when available (not `logsToTraceSteps`).
- Fixtures only when API unavailable or demo-mode true. Banner: “Fixture mode” vs “Live API mode.”
**Tests:** Manual; optional lightweight UI/type test.

---

## PHASE 5 — Non-financial agents useful and visible

### Commits 11–15 — Per-agent playbooks and artifacts
- **11** Graph Drift: 6–10 steps, drift report (summary_json or drift_reports), drift_warning risk_signal, “copy retrain command”; UI drift charts.
- **12** Evidence Narrative: evidence-only narrative, elder-safe version, caregiver escalation draft; persist narrative_reports; structured LLM output + citations.
- **13** Ring Discovery: NetworkX clustering + motif heuristics; persist rings, ring_candidate risk_signals; UI Ring page.
- **14** Continual Calibration: calibration curve per household, household_calibration + calibration report artifact; UI before/after and chart.
- **15** Synthetic Red-Team: scenario variants, pipeline per variant, regression report; persist replay_fixture; UI redteam report and “open in replay.”

**Global:** Each agent persists `agent_runs` with rich `step_trace`, at least one user-visible artifact, API endpoints, Agents page “View outputs”, dry_run preview.

---

## PHASE 6 — Consolidate graph building

### Commit 16 — domain/graph_service.build_graph_from_events
**Files:** `apps/api/domain/graph_service.py`, pipeline, worker, graph router
- Add `build_graph_from_events` wrapping GraphBuilder orchestration. Refactor pipeline `normalize_events`, graph router, worker graph builder to call shared helper. Single place for entity/mention/relationship persistence and RLS.
**Tests:** Shared helper unit tests; existing routes unchanged.

---

## Deliverables (after full implementation)

1. **This commit plan** — done in this file.
2. **Updated docs:** `docs/DEMO_MOMENTS.md` (real flows), `docs/GNN_PRODUCT_LOOP_AUDIT.md` (deleting GNN breaks demo).
3. **Cohesion checklist:** no unused important code paths; all agents produce visible artifacts; no fixtures masking live features outside demo mode.

---

## Implementation status

- **Phase 1:** Commit 1 (financial embedding persist), Commit 2 (frontend schema), Commit 3 (tests) — implemented.
- **Phase 2:** Commit 4 (centroid pattern `source` in pipeline), Commit 5 (worker creates `watchlist_embedding_match` risk_signal on match), Commit 6 (UI: watchlist provenance + alert detail “Matched centroid watchlist”) — implemented.
- **Phase 3:** Commit 7 (model_subgraph edges with `importance`; build_subgraph prefers `deep_dive_subgraph`), Commit 8 (POST `/risk_signals/{id}/explain/deep_dive?mode=pg|gnn`, persist `deep_dive_subgraph`; mode=gnn returns 501), Commit 9 (UI: graph evidence with PGExplainer / Deep dive toggle, “Compute deep dive” button) — implemented.
- **Phase 4:** Commit 10 (Replay: “Refresh from API” uses risk_signals + explanation subgraph for graph; step_trace when available; fixture only when API unavailable or demo mode; banner “Fixture mode” vs “Live API mode”) — implemented.
- **Phase 5:** Commit 11 (Graph Drift: `summary_json.retrain_command` when drift detected; Agents page “Copy retrain command” button for graph_drift when present) — implemented. Commit 12 (Evidence Narrative: persist `narrative_reports` with report_json; `caregiver_escalation_draft` in summary; GET `/agents/narrative/report/{id}`; Agents page “View report” link; `/reports/narrative/[id]` page) — implemented. Commit 13 (Ring Discovery: GET /rings, /rings/:id, Rings page and nav, alert “View ring” → /rings/:id; agent artifacts_refs) — implemented. Commit 14 (Continual Calibration: calibration_report in summary, GET /agents/calibration/report, “View calibration report”, /reports/calibration) — implemented. Commit 15 (Synthetic Red-Team: replay_payload in summary, GET /agents/redteam/report, “View red-team report”, /reports/redteam, “Open in replay” → /replay?source=redteam) — implemented.
- **Phase 6:** Commit 16 (domain/graph_service.build_graph_from_events wrapping GraphBuilder; optional persist when supabase provided; pipeline normalize_events, graph router _build_graph_from_events, worker run_graph_builder refactored to use it; tests/test_graph_service.py) — implemented.

---

## Cohesion checklist (post-implementation)

- **No unused important code paths:** Graph build is centralized in `domain.graph_service.build_graph_from_events`; pipeline, graph router, and worker call it. Agents use shared persist/step helpers.
- **All agents produce visible artifacts:** Financial (risk_signals, embeddings, watchlists). Graph Drift (summary + retrain_command, drift risk_signal). Evidence Narrative (narrative_reports, risk_signal updates, “View report”). Ring Discovery (rings, ring_members, ring_candidate signals, Rings page). Continual Calibration (household_calibration, calibration report, “View calibration report”). Synthetic Red-Team (replay_payload, “View red-team report”, “Open in replay”). Caregiver Outreach (outbound_actions, elder_safe_message).
- **No fixtures masking live features outside demo mode:** Similar incidents show “Unavailable” when backend says no embeddings. Replay uses API data when not in demo mode; banner indicates Fixture vs Live API vs Red-team. Agents page and report pages use live API; demo mode returns empty or fixture only where documented.
