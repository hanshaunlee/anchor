# Anchor Cohesion Plan — Commit-by-Commit

## Prioritized task list (with file paths and acceptance criteria)

### 1. RiskScoringResponse: add model_meta; pipeline set fallback_used
- **Files:** `apps/api/api/schemas.py`, `apps/api/domain/risk_scoring_service.py`, `apps/api/api/pipeline.py`
- **Criteria:** Response includes optional `model_meta: {model_name, checkpoint_id, embedding_dim}` when model ran. Pipeline sets `fallback_used="rule_only"` when using rule-only fallback. No placeholder embeddings.
- **Verify:** Unit test that when checkpoint missing, `score_risk` returns `model_available=False`, `scores=[]`; pipeline sets `fallback_used` and does not add embeddings to state.

### 2. Pipeline: explicit model_available in explanation; no stub subgraph when model off
- **Files:** `apps/api/api/pipeline.py`
- **Criteria:** When `_model_available` is False, explanations have `model_available: false`; no `model_subgraph` key when model didn't run. Rule-only fallback scores never include embedding or model_subgraph.
- **Verify:** Existing pipeline test; GNN product loop test.

### 3. Financial agent: persist step_trace in agent_runs
- **Files:** `apps/api/domain/agents/financial_security_agent.py`
- **Criteria:** Playbook builds `step_trace: [{step, status, ...}]` for ingest, normalize, detect, watchlist, persist; insert/update `agent_runs` includes `step_trace`. GET /agents/financial/trace returns it.
- **Verify:** Run financial agent, GET trace, assert step_trace present.

### 4. Evidence narrative: explanation.narrative + UI
- **Files:** `apps/api/domain/agents/evidence_narrative_agent.py`, `apps/web/src/app/(dashboard)/alerts/[id]/alert-detail-content.tsx`
- **Criteria:** Narrative agent writes caregiver narrative to `explanation.narrative` (restricted to evidence); UI shows narrative when present.
- **Verify:** Run narrative agent on signal with subgraph; GET risk signal detail; UI shows narrative.

### 5. Graph drift agent: real drift computation
- **Files:** `apps/api/domain/agents/graph_drift_agent.py`
- **Criteria:** Compare recent window (e.g. last 3 days) vs baseline (e.g. 7–14 days ago) embedding distribution; compute shift (e.g. mean distance or MMD); if shift > tau, open drift_warning risk_signal.
- **Verify:** Unit test with mock embeddings; drift > tau creates signal in dry_run=false.

### 6. Consent: redact at API response time
- **Files:** `apps/api/domain/risk_service.py`, `apps/api/api/routers/sessions.py` (if needed)
- **Criteria:** When household/session consent disallows sharing text: risk_signal detail response redacts raw utterance text in explanation and canonical values in subgraph; events already have text_redacted from DB. UI shows banner when redacted.
- **Verify:** With consent_share=false, GET detail and events; assert no raw text.

### 7. Replay: prefer API demo; document fixture
- **Files:** `apps/web/src/app/(dashboard)/replay/page.tsx`
- **Criteria:** Replay page uses GET /agents/financial/demo as primary source for scenario data when not in demo mode; fallback to fixture only for static demo; label "Load from API" as primary.
- **Verify:** Click "Load from API" populates timeline and trace from real demo response.

### 8. Tests: cohesion tests
- **Files:** `tests/test_gnn_product_loop.py`, new `tests/test_cohesion.py`
- **Criteria:** No checkpoint → similar incidents unavailable, centroid watchlist not created, model_subgraph absent. With checkpoint → similar works, model_subgraph has edges for at least one alert. E2E scenario: run demo script → pipeline → verify UI fixtures match backend artifacts.
- **Verify:** CI runs lint, unit tests, one pipeline integration smoke test.

### 9. Cohesion checklist and deploy
- **Files:** `docs/INTEGRATION_MAP.md` (checklist section), `docs/COHESION_PLAN.md`
- **Criteria:** Checklist confirms no unused agents/models, no placeholder outputs; deploy steps documented/run.

---

## Commit sequence (minimal diffs)

| # | Commit title | Files |
|---|---------------|--------|
| 1 | docs: add Integration Map and Cohesion Plan | docs/INTEGRATION_MAP.md, docs/COHESION_PLAN.md |
| 2 | api: add model_meta to RiskScoringResponse; set fallback_used in pipeline | apps/api/api/schemas.py, apps/api/domain/risk_scoring_service.py, apps/api/api/pipeline.py |
| 3 | api: financial agent persist step_trace in agent_runs | apps/api/domain/agents/financial_security_agent.py |
| 4 | api: evidence narrative agent write explanation.narrative | apps/api/domain/agents/evidence_narrative_agent.py |
| 5 | web: alert detail show explanation.narrative when present | apps/web/.../alerts/[id]/alert-detail-content.tsx |
| 6 | api: graph drift agent real two-window drift computation | apps/api/domain/agents/graph_drift_agent.py |
| 7 | api: consent-aware redaction in risk_signal detail response | apps/api/domain/risk_service.py |
| 8 | web: replay prefer API demo; clarify fixture fallback | apps/web/.../replay/page.tsx |
| 9 | tests: add cohesion tests (no checkpoint / with checkpoint) | tests/test_cohesion.py, tests/test_gnn_product_loop.py |
| 10 | docs: final cohesion checklist | docs/INTEGRATION_MAP.md |

---

## Schema/migration changes

- **risk_signals.explanation:** Already JSONB; add optional `narrative` (no migration if we just start writing it).
- **agent_runs.step_trace:** Already present (migration 006); ensure all agents write it.
- **RLS:** No change required for above.

---

## Deploy

1. **Backend (API)**  
   From repo root: install deps (e.g. `pip install -e .` with Python ≥3.11), set env (Supabase, optional NEO4J_URI, ANCHOR_ML_CHECKPOINT_PATH). Run `uvicorn api.main:app --reload` from `apps/api` or as configured.

2. **Worker**  
   Run worker jobs (cron or trigger); ensure `apps/worker` and `apps/api` are on PYTHONPATH.

3. **Web**  
   From `apps/web`: `npm install && npm run build && npm run start`. Set `NEXT_PUBLIC_API_BASE_URL` and Supabase env vars.

4. **Verify**  
   - GET /agents/status → agents list with last run.  
   - POST /agents/financial/run with dry_run=true → step_trace in response; after run, GET /agents/financial/trace?run_id=... → step_trace.  
   - GET /risk_signals/{id}/similar → available=false when no embedding.  
   - Replay page loads from API on mount; "Refresh from API" updates from demo run.
