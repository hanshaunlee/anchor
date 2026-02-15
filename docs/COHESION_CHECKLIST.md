# Cohesion Implementation Checklist (Original Prompt)

Verification against the original architectural and product-flow requirements.

---

## 0. Architectural Principles

| Requirement | Status | Notes |
|-------------|--------|--------|
| Financial + Narrative externally unified as Investigation | Done | POST /investigation/run runs both; narrative is HIDDEN in catalog |
| Scoring uses calibrated_p / fusion_score when available | Done | risk_scoring_service + financial agent use them; supervisor passes calibration_params |
| Conformal (q_hat) influences escalation consistently | Done | _escalation_triggered() in supervisor; outreach_candidates only when (1 - calibrated_p) >= q_hat when q_hat exists |
| Independence metadata flows through all investigation layers | Done | graph_service attaches to entities; financial explanation includes independence; narrative uses explanation |
| Drift + Calibration + Redteam orchestrated via Model Health agent | Done | model_health_agent.py runs drift, calibration, conformal validity, optional redteam |
| Outreach is Action system (calibrated severity + conformal + consent OR explicit caregiver) | Done | No "run outreach agent" as primary; POST /actions/outreach/preview and /send; auto_send only with consent + auto_send_outreach |
| Users never manually run drift, calibration, redteam | Done | They are ADVANCED_UI / Admin; NIGHTLY_MAINTENANCE runs model_health only |
| All agents runnable via Admin Tools | Done | ADMIN_BENCH + legacy POST /agents/*/run |

---

## 1. Agent Tiering + Registry

| Requirement | Status | Notes |
|-------------|--------|--------|
| domain/agents/registry.py | Done | AGENT_SPEC with tier, triggers, visibility |
| AgentTier = USER_ACTION \| INVESTIGATION \| SYSTEM_MAINTENANCE \| DEVTOOLS | Done | Constants in registry |
| AgentTrigger = MANUAL_UI \| AUTOMATIC \| SCHEDULED \| WEBHOOK \| ADMIN_ONLY | Done | |
| AgentVisibility = DEFAULT_UI \| ADVANCED_UI \| HIDDEN | Done | |
| required_roles, consent_requirements, requires_calibrated_model, requires_embeddings, affects_user_alerts, writes_embeddings | Done | Per-agent in AGENT_SPEC |
| financial → INVESTIGATION \| AUTOMATIC + MANUAL_UI \| DEFAULT_UI | Done | |
| narrative → INVESTIGATION \| AUTOMATIC \| HIDDEN | Done | |
| ring → INVESTIGATION \| AUTOMATIC (conditional) + ADMIN_ONLY \| ADVANCED_UI | Done | |
| outreach → USER_ACTION \| AUTOMATIC (gated) + MANUAL_UI \| DEFAULT_UI | Done | |
| device_high_risk_mode (if exists) | Partial | No separate agent; task type in incident_response; not in registry as standalone |
| model_health → SYSTEM_MAINTENANCE \| SCHEDULED + ADMIN_ONLY \| ADVANCED_UI | Done | |
| redteam → DEVTOOLS \| ADMIN_ONLY \| ADVANCED_UI | Done | |
| GET /agents/catalog filtered by role, consent, environment, calibration_present, model_available | Done | get_agents_catalog() + route |

---

## 2. Supervisor Orchestrator

| Requirement | Status | Notes |
|-------------|--------|--------|
| domain/agents/supervisor.py | Done | run_supervisor(), all modes |
| Slug "supervisor" | Done | SUPERVISOR_SLUG |
| run_mode: INGEST_PIPELINE \| NEW_ALERT \| NIGHTLY_MAINTENANCE \| ADMIN_BENCH | Done | |
| INGEST_PIPELINE: load_context, run_financial (calibrated_p + fusion_score, persist embeddings, structural_motifs + independence in explanation) | Done | financial playbook called with calibration_params; explanation includes structural_motifs, independence |
| ensure_narratives for each new/open signal (idempotent) | Done | run_evidence_narrative_agent(risk_signal_ids=...) |
| Severity >= threshold: conformal 1 - calibrated_p >= q_hat → escalate; create outreach_candidate (not send) | Done | _escalation_triggered(); outreach_candidates list; outbound_actions insert status=queued |
| Optional ring discovery (sufficient relationships OR structural motifs star/bridge) | Done | optional_ring_discovery step with condition |
| Persist supervisor agent_runs, child_run_ids | Done | persist_agent_run_ctx("supervisor", ...) |
| Return: new_signals, enriched_signals, outreach_candidates, watchlists, supervisor_run_id, child_run_ids | Done | created_signal_ids, outreach_candidates, summary_json.counts, step_trace |
| NEW_ALERT: ensure narrative, re-evaluate conformal, create/update outreach draft; auto_send if enabled + consent | Done | |
| NIGHTLY_MAINTENANCE: run model_health only; redteam only if env != prod or admin_force | Done | model_health_agent; redteam step conditional |
| ADMIN_BENCH: run force_agents, structured step_trace | Done | |

---

## 3. Investigation Flow

| Requirement | Status | Notes |
|-------------|--------|--------|
| POST /investigation/run | Done | investigation router |
| POST /alerts/{id}/refresh | Done | alerts router + risk_signals/{id}/refresh |
| Internally call supervisor INGEST_PIPELINE or NEW_ALERT | Done | |
| Narrative automatically enriched; structural motifs visible; independence in explanation; calibrated severity and decision_rule_used in UI | Done | API returns full explanation; UI not implemented |
| Users never manually run narrative | Done | HIDDEN in catalog; only via Investigation |

---

## 4. Model Health Consolidation

| Requirement | Status | Notes |
|-------------|--------|--------|
| domain/agents/model_health_agent.py | Done | |
| gather_artifacts, drift_check, calibration_check, conformal_validity_check, redteam_regression (optional), synthesis_report | Done | |
| Report: drift_detected, mmd_rbf, ks_stat, drift_confidence_interval, calibration_ece, conformal_coverage, recommendation | Done | summary_json |
| NIGHTLY_MAINTENANCE runs only model_health | Done | |
| Individual agents runnable via Admin | Done | POST /agents/drift/run etc. |

---

## 5. Outreach → Action System

| Requirement | Status | Notes |
|-------------|--------|--------|
| outbound_actions: risk_signal_id, channel, to, payload_json, provider, status, sent_at, conformal_triggered, calibrated_p_at_send, decision_rule_used | Done | Migration 015 adds columns; existing table had action_type, channel, etc. |
| household_capabilities.auto_send_outreach | Done | Migration 015 + capability_service |
| Workflow: Investigation produces outreach_candidate; UI "Notify caregiver"; preview + send endpoints | Done | POST /actions/outreach/preview, /send; GET /actions/outreach?risk_signal_id= |
| Auto-send only if calibrated severity high, conformal true, consent, auto_send_outreach true | Done | supervisor NEW_ALERT logic |
| Remove "Run outreach agent" from main Agents page | Doc only | UI not implemented; catalog marks outreach as USER_ACTION |

---

## 6. Trigger Wiring

| Requirement | Status | Notes |
|-------------|--------|--------|
| After ingest: worker calls supervisor INGEST_PIPELINE | Done | run_supervisor_ingest_pipeline() in worker/jobs.py |
| After signal persist: worker calls supervisor NEW_ALERT | Partial | Not wired in jobs.py; API POST /alerts/{id}/refresh exists for on-demand |
| Nightly: cron hits POST /system/maintenance/run | Done | Endpoint exists; cron config is deployment-specific |

---

## 7. UI Updates

| Requirement | Status | Notes |
|-------------|--------|--------|
| Dashboard: "Run Investigation"; latest run summary | Not done | API ready; web app not updated |
| Alert detail: calibrated score, decision rule, conformal flag, structural motifs, independence, Action Plan, outreach preview/send | Not done | API returns data; UI not implemented |
| Agents page: Actions, Advanced Tools, Model Health | Not done | GET /agents/catalog ready |
| Replay: use API step_trace/subgraph or label "Fixture demo" | Not done | |

---

## 8. Correctness Requirements

| Requirement | Status | Notes |
|-------------|--------|--------|
| Calibrated scores propagate everywhere | Done | financial + pipeline + supervisor |
| Conformal influences escalation consistently | Done | _escalation_triggered() |
| Narrative cites evidence IDs only | Done | evidence_only_guard in narrative agent |
| PGExplainer only via pg_service | Done | risk_scoring_service uses domain.explainers.pg_service |
| Embeddings persisted when model_available | Done | financial agent + worker |
| Timestamp utility everywhere | Done | time_utils used in pipeline, motifs; tests added |
| No duplicate explainer logic | Done | Single pg_service |
| No fake scores | Done | rule_only fallback explicit |
| Supervisor step_trace structured | Done | step() context manager |

---

## 9. Tests

| Test | Status | Notes |
|------|--------|--------|
| test_supervisor_ingest_runs_financial_then_narrative | Done | test_supervisor.py |
| test_conformal_escalation_path | Done | test_conformal_escalation.py |
| test_model_health_runs_all_components | Done | test_model_health_agent.py |
| test_outreach_auto_send_requires_conformal_and_consent | Done | test_supervisor.py (NEW_ALERT) |
| test_agents_catalog_respects_role | Done | test_agents_catalog.py |
| test_investigation_flow_persists_embeddings | Partial | Integration test; would need DB |
| test_structural_motifs_propagate_to_narrative | Done | test_structural_motifs_and_independence.py |
| test_supervisor_ingest_creates_outreach_draft_not_sent_by_default | Done | test_supervisor.py |
| test_supervisor_idempotent_second_run_does_not_duplicate_narratives_or_actions | Done | test_supervisor.py |
| test_conformal_trigger_logic_blocks/allows_outreach | Done | test_conformal_escalation.py |
| test_decision_rule_used_propagates | Done | test_conformal_escalation.py |
| test_model_unavailable_falls_back_to_rule_score_and_no_embeddings | Not added | Existing risk_scoring tests cover model_available=False |
| test_narrative_evidence_only_no_unreferenced_claims | Not added | evidence_only_guard exists in narrative agent |
| test_time_utils_accepts_datetime_iso_unix_and_handles_tz | Done | test_time_utils.py |
| test_independence_bridge_bonus_increases_rule_score | Not added | rule_scoring.compute_rule_score has bridge bonus; could add unit test |
| test_model_health_flags_stale_calibration_when_drift_high | Done | test_model_health_agent.py |
| stress_supervisor_matrix.py | Done | scripts/stress_supervisor_matrix.py |

---

## 10. Deliverables

| Deliverable | Status |
|------------|--------|
| registry.py | Done |
| supervisor.py | Done |
| model_health_agent.py | Done |
| outbound_actions schema + migration + endpoints | Done (migration 015; preview/send; GET ?risk_signal_id=) |
| investigation endpoints | Done |
| maintenance endpoints | Done |
| UI consolidation | Not done (API contract ready) |
| updated docs | Done (agents.md, this checklist) |
| updated tests | Done (new test files; existing kept passing) |
| Zero regressions | Intended (no removal of existing behavior) |

---

## Summary

- **Fully implemented:** Registry, Supervisor (all modes including optional_ring_discovery), Model Health agent, Investigation + Maintenance + Catalog + Actions endpoints, Financial agent (calibration + structural_motifs + independence), outbound_actions migration and capabilities, worker run_supervisor_ingest_pipeline, tests listed above, stress harness.
- **Partial / doc only:** device_high_risk_mode as separate registry entry (task type only); worker NEW_ALERT after each signal (API exists, not wired in job); narrative idempotency by explanation_version hash (narrative is effectively idempotent by re-updating).
- **Not implemented:** UI changes (Dashboard, Alert detail, Agents page, Replay); a few optional unit tests (model unavailable fallback, narrative evidence-only validator, independence bridge bonus); provider interface in domain/actions/outreach_providers.py (domain/notify/providers.py already provides Mock/Twilio/SendGrid).
