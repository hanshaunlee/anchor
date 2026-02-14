# Anchor: Codebase Critique & Ambitious Upgrade Plan

This document maps the critique and upgrade plan to the codebase and provides code-level deliverables. It extends the [GNN Product Loop Audit](./GNN_PRODUCT_LOOP_AUDIT.md).

---

## Implemented (summary)

- **Single risk scoring service:** `domain/risk_scoring_service.py` with `score_risk()` and `RiskScoringResponse`/`RiskScoreItem` in `api/schemas.py`. Pipeline, worker, and Financial Security Agent use it; explicit rule-only fallback when `model_available=false`.
- **Idempotent event ingestion:** `ingest_events()` uses `upsert(rows, on_conflict="session_id,seq")`; deterministic normalize (events sorted by `(ts, seq)` per session).
- **Explicit model_available (§1.3):** Risk signal detail and list card expose `model_available`; watchlist items have `model_available=True` for `embedding_centroid`; explanations always set `model_available` (explicit false when model didn’t run).
- **Similar incidents:** `retrieval_provenance` (model_name, checkpoint_id, embedding_dim, timestamp); **pgvector** migration `008_pgvector_embeddings.sql` (vector column, IVFFLAT cosine index, RPC `similar_incidents_by_vector`); similarity_service tries RPC then falls back to JSONB + Python cosine.
- **Embedding-centroid watchlists (§2.2):** `EmbeddingCentroidPattern` and `EmbeddingCentroidProvenance` in schemas; pipeline pattern includes `cosine_threshold`, `provenance.window_days`; worker server-side matching (hard dependency: no centroid when model didn’t run).
- **Model evidence subgraph (§2.3):** Stable DB IDs in `model_subgraph` (entity_id from `entities` in risk_scoring_service); `model_evidence_quality` (sparsity, edges_kept, edges_total) in explanation; `model_available=false` → no `model_subgraph`.
- **New agents (§4):** `graph_drift_agent`, `evidence_narrative_agent`, `ring_discovery_agent`, `continual_calibration_agent`, `synthetic_redteam_agent` with run functions and step_trace/summary_json; persisted to `agent_runs`.
- **Operational (§5):** `GET /agents/status` lists all KNOWN_AGENTS (financial_security, graph_drift, evidence_narrative, ring_discovery, continual_calibration, synthetic_redteam); `GET /agents/trace?run_id=&agent_name=` for any agent; `POST /agents/{drift,narrative,ring,calibration,redteam}/run` with dry_run; UI Agent Trace uses `step_trace` from API (friendly view in agents page).

---

## Executive summary

**Keep:** Event-sourced core, heterogeneous (HGT) modeling, embedding-first mentality, edge attributes as first-class.

**Refactor (non-negotiable):**

1. **Single risk scoring service** — One function + response schema used by pipeline, worker, and Financial Security Agent; no silent placeholder scores; explicit `model_available` in API payload.
2. **Idempotent event ingestion** — Keyed by `(session_id, seq)`; deterministic normalize step; aligns with LangGraph durable execution.
3. **Explicit `model_available`** — In risk signals, similar incidents, watchlists, explanations; fallbacks are deliberate modes, not silent.

**Make GNN drive product:** Similar incidents from real embeddings only; embedding-centroid watchlists hard-depend on embeddings; model-derived evidence subgraph with stable IDs and quality metric.

---

## 1. Refactor deliverables (code-level)

### 1.1 Single risk scoring service contract

**Current state:**

- **Pipeline** (`apps/api/api/pipeline.py`): `risk_score_inference()` — loads HGT from checkpoint, runs `ml.inference.run_inference`, attaches PGExplainer; on failure uses placeholder scores (no embeddings).
- **Worker** (`apps/worker/worker/jobs.py`): `run_risk_inference()` — **never calls ML**; returns placeholder list keyed by entity count.
- **Financial Security Agent** (`apps/api/domain/agents/financial_security_agent.py`): `_detect_risk_patterns()` — builds hetero data, optionally runs `run_inference`, blends motif + model; different payload shape (motif_rule_score, model_score, uncertainty).

**Deliverables:**

| Deliverable | Location | Change |
|-------------|----------|--------|
| Shared risk scoring API | New `domain/risk_scoring_service.py` (or `ml/risk_scoring.py`) | Single function `score_risk(household_id, graph_context) -> RiskScoringResponse` where `RiskScoringResponse` has `model_available: bool`, `scores: list[RiskScoreItem]`, optional `embeddings`, optional `model_subgraph` per item. No placeholder generation inside; if model unavailable, return `model_available=False` and empty or rule-only scores with explicit flag. |
| Pipeline | `pipeline.py` `risk_score_inference` | Call shared service; set `state["_model_available"]` from response; never build placeholder list in pipeline. |
| Worker | `jobs.py` `run_risk_inference` | Remove local placeholder loop; call same shared service (or call pipeline’s graph build + shared scorer). |
| Agent | `financial_security_agent.py` `_detect_risk_patterns` | Call shared risk scoring service for GNN path; keep motif layer as separate “rule” layer; merge in agent with explicit `model_available` in each risk item. |
| Schema | `api/schemas.py` | Add `RiskScoringResponse`, `RiskScoreItem` with `model_available`, `embedding` optional, `model_subgraph` optional. |

**Acceptance:** No code path generates placeholder scores/embeddings silently; any fallback is explicit in the API payload (e.g. `model_available=false`).

---

### 1.2 Idempotent event ingestion + deterministic normalize

**Current state:**

- **DB:** `events` has `UNIQUE (session_id, seq)` (`db/bootstrap_supabase.sql`, `001_initial_schema.sql`).
- **Ingest** (`apps/api/domain/ingest_service.py`): `ingest_events()` does plain `insert(rows)`; duplicate `(session_id, seq)` causes DB error.
- **Pipeline** (`pipeline.py`): `normalize_events(state)` uses GraphBuilder; no deduplication by `(session_id, seq)` before normalize.

**Deliverables:**

| Deliverable | Location | Change |
|-------------|----------|--------|
| Idempotent insert | `domain/ingest_service.py` | Use `INSERT ... ON CONFLICT (session_id, seq) DO UPDATE SET ...` (or `DO NOTHING`) so re-sending same event batch is safe. Optionally set `ingested_at = now()` on conflict. |
| Deterministic normalize | `pipeline.py` / `domain/graph_service.py` | Ensure `normalize_events` is deterministic: e.g. sort events by `(session_id, seq)` before processing; no reliance on dict iteration order for entities/mentions. Document that side effects (DB writes) are outside this step when used in LangGraph. |
| Worker ingest | `worker/jobs.py` | When fetching events for pipeline, rely on same (session_id, seq) semantics; no duplicate events inserted if worker retries. |

**Acceptance:** Same `(session_id, seq)` can be ingested twice without error; normalize output is deterministic for same input.

---

### 1.3 Explicit `model_available` signaling

**Current state:**

- Pipeline sets `_model_available` and explanation has `model_available`; risk_signal detail gets `explanation` from DB.
- Similar incidents: `SimilarIncidentsResponse.available` and `reason="model_not_run"` already exist.
- Watchlists: embedding_centroid watchlist only created when embeddings exist (no explicit “model_available” on watchlist item).

**Deliverables:**

| Deliverable | Location | Change |
|-------------|----------|--------|
| Risk signal list/detail | `domain/risk_service.py`, `schemas.py` | Ensure `explanation` (or top-level) includes `model_available` so UI can show “Model unavailable” when false. |
| Similar incidents | `domain/similarity_service.py`, `schemas.py` | Keep `available` + `reason`; add optional `retrieval_provenance` (see §2.1). |
| Watchlists | When returning watchlist items | For `watch_type=embedding_centroid`, include `model_available: true` (or provenance); when embeddings missing, centroid watchlists do not exist (already true; document as contract). |
| Explanations | `generate_explanations` | Already set `model_available` in explanation_json; ensure it is never omitted when model didn’t run (explicit false). |

**Acceptance:** Every surface that depends on the GNN (risk scores, similar incidents, watchlists, explanations) exposes a clear model_available (or equivalent) so fallbacks are deliberate.

---

## 2. Make the GNN drive product surfaces

### 2.1 Similar incidents from real embeddings

**Current state:** `domain/similarity_service.py`: uses `risk_signal_embeddings` with `has_embedding`; cosine in Python over JSONB embedding; returns `available=False, reason="model_not_run"` when no embedding.

**Deliverables:**

| Deliverable | Location | Change |
|-------------|----------|--------|
| pgvector | New migration | Add `vector` column (e.g. `vector real[]` or pgvector type) to `risk_signal_embeddings`; IVFFLAT index with `vector_cosine_ops`; backfill from existing `embedding` JSONB. |
| Similar endpoint | `routers/risk_signals.py`, `similarity_service.py` | `GET /risk_signals/{id}/similar`: return `available=false` if no embedding row or `has_embedding=false`; else query by cosine distance (pgvector); support `top_k`, `window_days`; add response field `retrieval_provenance: { model_name, checkpoint_id, embedding_dim, timestamp }` from embedding row. |
| No fabrication | Already satisfied | Do not create or return synthetic vectors when model didn’t run. |

**Acceptance:**

- If model unavailable → Similar Incidents unavailable (not “meaningless but working”).
- If model available → top neighbors by cosine; provenance in response; optional test: neighbors correlate with same motif category more often than random on synthetic scenarios.

---

### 2.2 Embedding-centroid watchlists

**Current state:** Pipeline `_embedding_centroid_watchlist()` builds centroid from high-risk embeddings; pattern has `centroid`, `threshold`, `metric`, `provenance` (risk_signal_ids empty, node_indices set). Worker `_check_embedding_centroid_watchlists()` matches new embeddings against active centroid watchlists and updates risk_signal explanation.

**Deliverables:**

| Deliverable | Location | Change |
|-------------|----------|--------|
| Pattern schema | `pipeline.py`, `worker/jobs.py`, `schemas.py` | Formalize `watch_type="embedding_centroid"` pattern: `centroid` (L2-normalized vector), `cosine_threshold`, `provenance: { risk_signal_ids, window }`; store in DB (pattern JSON or pgvector for centroid). |
| pgvector for centroids | Optional | Store centroid in pgvector for server-side similarity; or compute in Python and store in `pattern.centroid`; matching loop uses same cosine threshold. |
| Server-side matching | `worker/jobs.py` | When new risk embeddings arrive, compute similarity to active centroid watchlists; if above threshold, create risk_signal or escalate; “hard dependency”: if embeddings missing, centroid watchlists do not exist (hash/keyword watchlists still work). |

**Acceptance:** Centroid watchlist pattern is documented and used consistently; no centroid watchlist when model didn’t run.

---

### 2.3 Model-derived evidence subgraph

**Current state:** Pipeline `_attach_pg_explainer_subgraphs()` uses PGExplainer (when available), writes `model_subgraph` with nodes/edges keyed by **PyG node index** (`id: str(node_idx)`). Financial Security Agent builds `evidence_subgraph` from entity list (DB entity ids) but `model_subgraph` is stub (nodes only, edges []). `explain_service.build_subgraph_from_explanation()` uses `explanation.get("subgraph") or explanation.get("model_subgraph")`.

**Deliverables:**

| Deliverable | Location | Change |
|-------------|----------|--------|
| Stable ID mapping | Pipeline + explain_service | When attaching `model_subgraph`, map PyG node indices to `entity_id` / `event_id` (DB IDs) using state’s entities/events; API and UI receive only DB IDs. |
| PGExplainer / GNNExplainer | `pipeline.py`, `ml/explainers/` | Keep PGExplainer for scalable inductive use; optionally GNNExplainer for deep-dive; optional SubgraphX for “investigation mode” (offline). |
| Evidence quality metric | Explanation schema | Add `model_evidence_quality`: `sparsity` (% edges kept), optional `fidelity_proxy` (prediction drop when edges removed). |
| model_available=false | `generate_explanations` | When `model_available=false`, do not include `model_subgraph`; UI shows “Model unavailable.” |

**Acceptance:**

- When `model_available=true`, evidence subgraph has >0 edges (when graph has edges) and changes with seed node; nodes/edges use DB IDs.
- When `model_available=false`, `model_subgraph` absent; UI shows “Model unavailable.”

---

## 3. Research-based model upgrades (roadmap)

| Area | Reference | Deliverable (feasible) |
|------|-----------|-------------------------|
| Temporal graph | TGN-style memory | TGN-style memory module for continuous-time event graph; time encodings (e.g. TGAT Δt) in attention/edge bias. |
| Self-supervised pretraining | DGI / GraphCL | Deep Graph Infomax or GraphCL augmentations for label-free embeddings; optional CoLA-style contrastive anomaly head. |
| Scalable training | GraphSAINT | Subgraph sampling instead of full-graph training for large households. |
| Uncertainty & calibration | Temperature scaling, conformal | Calibrate probabilities (temperature scaling); split conformal prediction sets for GNN to expose confidence guarantees. |

---

## 4. New agents (exact deliverables)

| Agent | Research basis | Deliverables |
|-------|----------------|--------------|
| **Graph Drift Agent** | Embedding drift + calibration | Nightly job: embedding distribution shift; open `risk_signal` with type `drift_warning` if shift > τ; trigger retrain suggestion (e.g. Modal). |
| **Evidence Narrative Agent** | Explanation as subgraph + summarization | Input: model_subgraph + motifs → caregiver-readable narrative grounded in evidence pointers; store in `risk_signals.explanation.summary`. |
| **Ring Discovery Agent** | Community/similarity | Neo4j GDS node similarity + embeddings (FastRP/Node2Vec) on mirrored evidence graph; flag suspicious clusters; link to risk signals. |
| **Continual Calibration Agent** | Temperature scaling + conformal | Update household calibration from feedback; optional weekly recalibrate classifier head; produce calibration report. |
| **Synthetic Red-Team Agent** | Contrastive/anomaly eval | Generate new scam variants for replay; validate Similar Incidents + centroid watchlists (regression suite). |

Orchestration: LangGraph nodes with durable checkpoints; each agent is a node or subgraph.

---

## 5. Operational deliverables

| Deliverable | Location | Change |
|-------------|----------|--------|
| Router prefix `/agents/*` | `routers/agents.py` | Already under `/agents`; add (or document) status, last run, dry-run preview for each agent type. |
| Agent trace in UI | `agent_runs.step_trace` | Expose `GET /agents/{agent}/trace?run_id=...` (or under existing `/agents/financial/trace`); UI shows “Agent Trace” (friendly view of step_trace, not raw logs). |

---

## 6. File map (quick reference)

| Concern | Primary files |
|---------|----------------|
| Risk scoring | `api/pipeline.py` (risk_score_inference), `worker/jobs.py` (run_risk_inference), `domain/agents/financial_security_agent.py` (_detect_risk_patterns), `ml/inference.py` |
| Event ingest | `domain/ingest_service.py`, `db/bootstrap_supabase.sql` (events UNIQUE), `pipeline.py` (ingest_events_batch, normalize_events) |
| Similar incidents | `domain/similarity_service.py`, `api/routers/risk_signals.py`, `api/schemas.py` (SimilarIncidentsResponse), `db/migrations/003_*, 007_*` |
| Watchlists | `api/pipeline.py` (_embedding_centroid_watchlist, synthesize_watchlists), `worker/jobs.py` (_check_embedding_centroid_watchlists), `domain/watchlist_service.py` |
| Explanations / model_subgraph | `api/pipeline.py` (_attach_pg_explainer_subgraphs, generate_explanations), `domain/explain_service.py`, `domain/risk_service.py` |
| Agents | `api/routers/agents.py`, `domain/agents/financial_security_agent.py`, DB `agent_runs` (step_trace) |

---

## 7. Implementation order (suggested)

1. **Single risk scoring service** — Unify pipeline, worker, agent on one contract; explicit `model_available` in response.
2. **Idempotent event ingestion** — ON CONFLICT on (session_id, seq); deterministic normalize.
3. **Similar incidents** — Add retrieval_provenance; optionally migrate to pgvector (IVFFLAT).
4. **Embedding-centroid watchlist** — Formalize pattern schema; document hard dependency; server-side matching (already partially in worker).
5. **Model evidence subgraph** — Stable DB IDs in model_subgraph; evidence quality metric; ensure model_available drives visibility.
6. **New agents** — Implement one at a time (e.g. Evidence Narrative, then Graph Drift); expose status + trace under `/agents/*`.
7. **Research upgrades** — TGN, DGI/GraphCL, GraphSAINT, calibration as separate tracks.

This order makes the product loop depend on a single, explicit risk-scoring contract and removes silent fallbacks before adding pgvector and new agents.
