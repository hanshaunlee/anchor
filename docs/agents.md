# Anchor Agents

Anchor agents are ordered playbooks that run on household data. They use a shared framework (`domain/agents/base`: `AgentContext`, `step()`, persist helpers) and optionally the shared risk scoring service and ML artifacts. All support **dry-run** (preview without DB write) and persist `step_trace` and `summary_json` to `agent_runs`.

| Agent | Slug | Purpose | API |
|-------|------|---------|-----|
| Financial Security | `financial` | Ingest → risk patterns → recommendations → watchlist; read-only, no money movement | `POST /agents/financial/run`, `GET /agents/financial/demo`, `GET /agents/financial/trace?run_id=` |
| Graph Drift | `drift` | Multi-metric embedding drift; root-cause; `drift_warning` risk_signal | `POST /agents/drift/run` |
| Evidence Narrative | `narrative` | Evidence-grounded narrative for open signals; redaction-aware | `POST /agents/narrative/run` |
| Ring Discovery | `ring` | Interaction graph clustering; `ring_candidate` risk_signals; rings/connectors | `POST /agents/ring/run` |
| Continual Calibration | `calibration` | Platt/conformal from feedback; household_calibration; ECE report | `POST /agents/calibration/run` |
| Synthetic Red-Team | `redteam` | Scenario DSL + regression harness; pass rate, failing_cases | `POST /agents/redteam/run` |
| Caregiver Outreach | `outreach` | Outbound notify/call/email to caregiver; consent-gated; evidence bundle + elder_safe message | `POST /agents/outreach/run`, `POST /actions/outreach` |

- **Status:** `GET /agents/status` — last run per agent (`last_run_at`, `last_run_status`, `last_run_summary`).
- **Trace:** `GET /agents/trace?run_id=&agent_name=` or `GET /agents/{slug}/trace?run_id=` — step_trace and summary for any run.

---

## Financial Security Agent

The **Financial Security Agent** runs an ordered “financial protection playbook” to protect an elder user. It is **read-only** in terms of external actions: it can recommend, draft, and flag; it **cannot move money** or execute financial transactions.

### Ordered playbook (tasks)

1. **Ingest & normalize**
   - Pull recent events for the household in a configurable time window (default 7 days) from Supabase, or use pre-filled events from the LangGraph state.
   - Normalize to a consistent internal representation using the existing GraphBuilder: utterances, intents, entities, mentions, relationships.

2. **Detect risk patterns (rule + model)**
   - **Motif/rule layer**: scam-like patterns (e.g. “urgency + sensitive info request”, “new unknown contact”, “repeat attempts”, “new payee attempt” when available).
   - **Model layer** (optional): GNN risk scoring (HGT/GraphGPS/FraudGT-style) on entity subgraphs when a checkpoint is available.
   - Output: calibrated risk score, severity (1–5), and uncertainty.

3. **Investigation package**
   - Evidence bundle for the UI:
     - `motif_tags`
     - `timeline_snippet` (3–6 key events)
     - Evidence subgraph (nodes/edges with importance)
     - “What changed vs baseline” summary (e.g. new contact first seen, spike in attempts).
   - Uses existing explainers (motifs + PGExplainer/GNNExplainer) when available.

4. **Protective recommendations (non-destructive)**
   - `recommended_action` JSON with a tailored checklist, e.g.:
     - “Call back using saved contact”
     - “Do not share OTP/codes”
     - “Freeze unknown caller interactions for 60 minutes” (recommendation only)
     - “Enable bank alerts”
     - “Change passwords / 2FA”
     - “Verify payee”
     - “Review recent transactions and confirm unknown merchants” (when applicable)

5. **Watchlist synthesis**
   - Converts risk into compact watchlist items for the edge device: phone/email/name hashes, risky topic keywords.
   - Inserts into the `watchlists` table with priority and expiry (when not in dry-run and consent allows).

6. **Consent-gated escalation draft**
   - If severity ≥ threshold **and** `consent_state.share_with_caregiver === true`: draft an escalation message (stored in `recommended_action` or explanation; **not sent**).
   - Otherwise: no escalation payload is persisted beyond the elder’s allowed scope. If model confidence is low, the agent recommends a clarification question instead of escalating.

7. **Persist + notify**
   - Inserts/updates `risk_signals` with:
     - `signal_type` (e.g. `possible_scam_contact`, `social_engineering_risk`, `payment_anomaly`)
     - `severity` (1–5), `score`, `explanation` (motifs, subgraph, timeline), `recommended_action`, `status = open`
   - Broadcasts new alerts over WebSocket `/ws/risk_signals` when run via the API with persist enabled.

### Integration

- **LangGraph**: The node `financial_security_agent` runs after `graph_update` and before `risk_score`. It uses state (utterances, entities, mentions, relationships); when run inside the pipeline it does **not** persist to the DB (no Supabase in context). Use **POST /agents/financial/run** to run with persist and broadcast.
- **On-demand API**: **POST /agents/financial/run** with optional body `{ household_id?, time_window_days?, dry_run? }`. If `dry_run=true`, no DB write; response includes the computed risk_signals and watchlists for preview.
- **Consent**: Respects `sessions.consent_state` (e.g. `share_with_caregiver`, `watchlist_ok`). Does not expose redacted text when consent disallows.

### API usage

- **Run agent (persist or preview)**  
  `POST /agents/financial/run`  
  Body: `{ "time_window_days": 7, "dry_run": false }`  
  Returns: `run_id`, `risk_signals_count`, `watchlists_count`, `inserted_signal_ids`, `logs`; when `dry_run=true` also `risk_signals` and `watchlists`.

- **Agent status**  
  `GET /agents/status`  
  Returns: list of agents with `agent_name`, `last_run_at`, `last_run_status`, `last_run_summary`.

- **Financial run trace**  
  `GET /agents/financial/trace?run_id=<uuid>`  
  Returns: one row from `agent_runs` for that financial run.

### Testing the Financial Security Agent

**1. Demo via API (no auth) — see input + output**

```bash
curl -s http://127.0.0.1:8000/agents/financial/demo | jq
```

Returns `input_events` (3 demo events: Medicare urgency + share_ssn + phone) and `output` (logs, motif_tags, risk_signals, watchlists). No DB write.

**2. Via API with auth (optional demo events)**

- **Preview with demo events (no DB write, full input/output in response):**  
  `POST /agents/financial/run`  
  Body: `{ "dry_run": true, "use_demo_events": true }`  
  Header: `Authorization: Bearer <your_supabase_jwt>`  
  Response includes `input_events` and `risk_signals`, `watchlists`, `motif_tags`, `timeline_snippet`.
- **Live run on DB events:**  
  Body: `{ "dry_run": false }` — agent fetches events for your household from the last 7 days.
- **Live run on demo events (persist demo output to your household):**  
  Body: `{ "dry_run": false, "use_demo_events": true }`
- **Check status:**  
  `GET /agents/status`  
  `GET /agents/financial/trace?run_id=<uuid>`

**3. Unit tests (no API, no DB)**

```bash
pytest tests/test_financial_agent.py -v
```

**4. One-command demo harness (recommended for judges)**

```bash
PYTHONPATH=".:apps/api" python3 scripts/demo_replay.py
```

Writes `demo_out/risk_chart.json`, `explanation_subgraph.json`, `agent_trace.json`, and `scenario_replay.json`. Use `--ui` to copy the replay payload into the web app fixtures; use `--launch-ui` to also start the Next.js dev server in demo mode.

**5. Local script (no API, playbook only)**

```bash
PYTHONPATH=".:apps/api" python3 scripts/run_financial_agent_demo.py
```

**6. As part of the LangGraph pipeline**

The pipeline runs the financial agent as a node after `graph_update`. It only updates in-memory state (no persist). To persist financial outputs, run the agent via **POST /agents/financial/run** (e.g. after ingest or on a schedule).

### Safety / policy

- Does not recommend illegal actions.
- Does not ask for or store raw bank credentials.
- Stores hashes/metadata where possible (e.g. watchlist patterns).
- If consent disallows, sensitive text in explanation payload is redacted.
- If model confidence is low, recommends a clarification question instead of escalating.

---

## Graph Drift Agent

The **Graph Drift Agent** detects distribution shift in risk-signal embeddings over time and produces a root-cause classification and optional action plan. It is **read-only** for external systems: it writes `risk_signals` (e.g. `drift_warning`), `summaries`, and `agent_runs` only.

### Ordered playbook (steps)

1. **intake_scope** — Set time windows: recent (e.g. 3 days) vs baseline (e.g. 14 days ending 7 days ago); require minimum samples per window.
2. **data_collection** — Fetch embeddings from `risk_signal_embeddings` for the household in each window (via `domain/ml_artifacts.fetch_embeddings_window`).
3. **quality_checks** — Ensure enough samples in both windows; otherwise report “insufficient samples” and skip drift decision.
4. **drift_metrics** — Compute multi-metric drift: centroid shift (1 − cosine_sim), MMD/energy distance, PCA+KS aggregate, neighbor stability (k-NN). Composite decision uses configurable threshold (e.g. τ = 0.15).
5. **slice_analysis** — Optional slice-level view (e.g. by signal_type or time slice).
6. **prototype_extraction** — Top-k prototype examples from recent window (e.g. 3) for explainability.
7. **root_cause_classification** — Optional LLM or rule: classify cause as `model_change`, `new_pattern`, or `behavior_shift`.
8. **action_plan_artifacts** — Build action summary and artifacts (e.g. “recalibrate” or “review new patterns”).
9. **persist_notify** — If drift &gt; threshold: upsert `drift_warning` risk_signal; upsert summary; persist `agent_runs` with step_trace and summary_json (e.g. `drift_detected`, `metrics`, `cause`, `examples`).

### Integration and API

- **On-demand only:** No pipeline node. Run via **POST /agents/drift/run** with body `{ "dry_run": true }` (default) or `{ "dry_run": false }` to persist.
- **Response:** `run_id`, `step_trace`, `summary_json` (metrics, drift_detected, cause, examples).
- **Trace:** `GET /agents/drift/trace?run_id=` or `GET /agents/trace?run_id=&agent_name=graph_drift`.

### Configuration (agent_settings)

- `drift_window_recent_days`, `drift_window_baseline_days`, `drift_baseline_end_days`, `drift_min_samples_per_window`, `drift_threshold`, `drift_top_prototype_examples`, `drift_k_neighbors`, `drift_pca_components`, `drift_mmd_threshold`, `drift_ks_threshold`, `drift_severity_high_centroid`.

---

## Evidence Narrative Agent

The **Evidence Narrative Agent** (“Investigation Packager”) turns open risk signals into evidence-grounded narratives for the UI: caregiver-facing narrative, optional hypotheses, and elder-safe version. It respects **consent**: text and entity canonicals can be redacted when consent disallows.

### Ordered playbook (steps)

1. **fetch_signals** — Select open risk signals (optionally filtered by `risk_signal_ids` or single `risk_signal_id`); fetch evidence (explanations, subgraphs, timeline snippets) from DB.
2. **evidence_normalization** — Normalize to canonical EvidenceBundle: nodes (id, type, label_or_hash, importance, first_seen, last_seen), edges (src, dst, type, importance), timeline; apply redaction (timeline text, entity canonical) per consent.
3. **what_changed_diff** — Compute “what changed vs baseline” (e.g. new contact, spike in attempts) for the narrative.
4. **hypothesis_generation** — Optional LLM step: generate short hypotheses from evidence.
5. **caregiver_narrative** — Optional LLM step: generate caregiver-facing narrative from evidence + hypotheses.
6. **elder_safe_version** — Produce elder-safe summary (no sensitive detail); may set `narrative_evidence_only` badge when only evidence (no LLM) is used.
7. **persist_risk_signals_summaries** — Update `risk_signals.explanation` (summary, narrative, narrative_evidence_only) and upsert `summaries` as needed.
8. **ui_integration** — Prepare payload for UI (narrative, hypotheses, elder_safe, evidence pointers).

### Integration and API

- **On-demand only:** **POST /agents/narrative/run** with body `{ "dry_run": true }` (default) or `{ "dry_run": false }`. Household from auth.
- **Response:** `run_id`, `step_trace`, `summary_json`.
- **Persistence:** When not dry_run, can persist to **narrative_reports** (migration 014) for “View report” in the UI (report_json: caregiver narrative, elder_safe, hypotheses).
- **Consent:** Reads `sessions.consent_state` (e.g. `share_with_caregiver`, text/entity sharing); redacts timeline snippets and entity display when disallowed.
- **Trace:** `GET /agents/narrative/trace?run_id=` or `GET /agents/trace?run_id=&agent_name=evidence_narrative`.

---

## Ring Discovery Agent

The **Ring Discovery Agent** finds “rings” (tight clusters of entities) and connectors in the household interaction graph, and emits `ring_candidate` risk_signals plus watchlists. It can use **NetworkX** (default) or **Neo4j GDS** when enabled.

### Ordered playbook (steps)

1. **data_acquisition** — Load relationships and mentions from Supabase (or Neo4j) for the household in a lookback window.
2. **build_graph** — Build interaction graph: nodes = entities, edges = relationship/mention with weights.
3. **candidate_ring_discovery** — Community detection (e.g. Louvain or label propagation); filter by min community size; take top rings.
4. **ring_scoring** — Score rings (e.g. novelty, size, activity).
5. **connector_bridge_analysis** — Identify bridge/connector entities between rings.
6. **evidence_subgraph_per_ring** — Build evidence subgraph per ring for explanations.
7. **output_artifacts** — Assemble ring list with members, connectors, evidence.
8. **watchlists_derived** — Derive watchlist entries from ring members/connectors (hashes, priorities).
9. **escalation_draft** — Optional escalation draft for high-priority rings (consent-gated).
10. **persist_ui** — Upsert `rings`, `ring_members` (migration 009); upsert `ring_candidate` risk_signals; persist watchlists when not dry_run; persist `agent_runs`.

### Integration and API

- **On-demand only:** **POST /agents/ring/run** with body `{ "dry_run": true }` (default) or `{ "dry_run": false }`. Optional `neo4j_available=False` in code (API uses server config).
- **Response:** `run_id`, `step_trace`, `summary_json` (e.g. rings_found, ring_candidate count).
- **Trace:** `GET /agents/ring/trace?run_id=` or `GET /agents/trace?run_id=&agent_name=ring_discovery`.

### Configuration (agent_settings)

- `ring_min_community_size`, `ring_top_rings`, `ring_novelty_days`, `ring_lookback_days`.

---

## Continual Calibration Agent

The **Continual Calibration Agent** refits score-to-probability calibration (e.g. Platt scaling or conformal thresholds) from user feedback and updates household-level calibration for the risk scoring pipeline.

### Ordered playbook (steps)

1. **gather_labeled_data** — Join `feedback` (true_positive / false_positive) with `risk_signals` to get (score, label, signal_type).
2. **data_quality_bias** — Check minimum labeled count; optional bias/slice checks.
3. **choose_method** — Select calibration method (Platt vs conformal) based on data size and config.
4. **fit_calibration** — Fit Platt (sigmoid) or conformal threshold on (score, label).
5. **update_household_calibration** — Write `household_calibration` (e.g. `calibration_params`, `last_calibrated_at` — migration 010).
6. **policy_recommendations** — Optional “policy patch” recommendations (e.g. raise/lower threshold by signal_type).
7. **calibration_report_artifact** — Before/after ECE (expected calibration error) or reliability diagram data.
8. **ui_summary** — Summary for UI (e.g. calibration_chart, policy_patch).

### Integration and API

- **On-demand only:** **POST /agents/calibration/run** with body `{ "dry_run": true }` (default) or `{ "dry_run": false }`.
- **Response:** `run_id`, `step_trace`, `summary_json`.
- **Trace:** `GET /agents/calibration/trace?run_id=` or `GET /agents/trace?run_id=&agent_name=continual_calibration`.

### Configuration (agent_settings)

- `calibration_min_labeled`, `calibration_target_fpr`, `calibration_ece_bins`.

---

## Synthetic Red-Team Agent

The **Synthetic Red-Team Agent** generates scenario variants from a DSL (themes: medicare, irs, grandchild, bank_fraud, crypto; tactics: urgency, authority, otp_request, etc.), runs a sandbox pipeline on them, and runs regression checks (similar incidents, centroid watchlist, evidence subgraph). Results are reported as pass rate and failing_cases.

### Ordered playbook (steps)

1. **define_objectives** — Set regression objectives (themes, number of variants).
2. **generate_variants** — Generate scenario DSL variants (paraphrases, theme/tactic combos).
3. **run_regression** — Run pipeline (or risk scoring) in sandbox for each scenario.
4. **similar_incidents_regression** — Assert similar-incidents retrieval (e.g. threshold on similarity).
5. **centroid_watchlist_regression** — Assert embedding-centroid watchlist behavior (e.g. new scenario near known centroid).
6. **evidence_subgraph_regression** — Assert evidence subgraph (e.g. min edges, expected edge types).
7. **generate_replay_artifact** — Build replay fixture (e.g. for demo or UI).
8. **user_visible_report** — Pass rate, failing_cases, scenarios_generated for summary_json.
9. **persist_ui** — Persist `agent_runs`; optionally risk_signals or replay fixture when not dry_run.

### Integration and API

- **On-demand only:** **POST /agents/redteam/run** with body `{ "dry_run": true }` (default) or `{ "dry_run": false }`.
- **Response:** `run_id`, `step_trace`, `summary_json` (e.g. regression_pass_rate, failing_cases, scenarios_generated).
- **Trace:** `GET /agents/redteam/trace?run_id=` or `GET /agents/trace?run_id=&agent_name=synthetic_redteam`.

### Configuration

- Scenario themes and tactics are in code (`SCENARIO_THEMES`, `TACTICS`, `SYNONYM_MAP`); similarity and pass-rate thresholds configurable.
