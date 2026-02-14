# Anchor API contracts for Web + Mobile UI

Auth: Supabase Auth. Roles: `elder`, `caregiver`, `admin`. All access is household-scoped (RLS).

---

## Auth

- **Sign in**: Supabase Auth (email/password or OTP). JWT in `Authorization: Bearer <token>`.
- **Household**: After sign-in, call `GET /households/me` to get `household_id`, `role`, `display_name`.

---

## REST endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /households/me | Household metadata + user role |
| POST | /households/onboard | Create household + link user (after sign-up); body: display_name?, household_name? |
| GET | /sessions?from=&to= | List sessions (id, device_id, timestamps, consent_state, summary) |
| GET | /sessions/{id}/events | Paginated events (redacted by consent) |
| GET | /risk_signals?status=&severity>= | List risk signals |
| GET | /risk_signals/{id} | Full detail + explanation_json + evidence pointers |
| GET | /risk_signals/{id}/similar?top_k= | Similar incidents by embedding (cosine). Returns `available`, `reason?`, `similar`; when no embedding: `available: false`, `reason: "model_not_run"`, `similar: []`. |
| POST | /risk_signals/{id}/feedback | Submit label (true_positive / false_positive / unsure) + notes |
| GET | /watchlists | Watchlists for device + UI |
| POST | /device/sync | Device heartbeat; returns watchlists delta + last upload pointers |
| POST | /ingest/events | Batch ingest events (device authenticated) |
| GET | /summaries?from=&to= | Weekly rollups / session summaries |
| GET | /agents/financial/demo | Run agent on demo events (no auth); returns input_events + output for inspection |
| POST | /agents/financial/run | Run Financial Security Agent (body: household_id?, time_window_days?, dry_run?, use_demo_events?) |
| POST | /agents/drift/run | Graph Drift Agent (body: dry_run?). Returns run_id, step_trace, summary_json (metrics, drift_detected, cause, examples) |
| POST | /agents/narrative/run | Evidence Narrative Agent (body: dry_run?). Evidence-grounded narrative for open signals |
| POST | /agents/ring/run | Ring Discovery Agent (body: dry_run?). Interaction graph clustering; ring_candidate risk_signals |
| POST | /agents/calibration/run | Continual Calibration Agent (body: dry_run?). Platt/conformal from feedback |
| POST | /agents/redteam/run | Synthetic Red-Team Agent (body: dry_run?). Scenario DSL + regression; pass rate, failing_cases |
| POST | /agents/outreach/run | Caregiver Outreach Agent (body: risk_signal_id, dry_run?). Caregiver/admin only. Returns run_id, outbound_action_id, step_trace, summary_json, suppressed, sent |
| GET | /agents/status | List agents and last run time / status / summary (from registry + last agent_runs) |
| GET | /agents/trace?run_id=&agent_name= | Get trace for any agent run |
| GET | /agents/{agent_slug}/trace?run_id= | Get trace by slug (e.g. drift, narrative, ring, calibration, redteam, financial, outreach) |
| GET | /agents/financial/trace?run_id= | Get trace for a financial agent run |
| POST | /actions/outreach | Trigger caregiver outreach for a risk signal (body: risk_signal_id, channel_preference?, dry_run?). Caregiver/admin only. dry_run returns preview only, no send. Returns ok, outbound_action?, agent_run_id, preview?, suppressed, sent |
| GET | /actions/outreach | List recent outbound actions for household (query: household_id?, limit?). Elder sees elder_safe_message only |
| GET | /actions/outreach/{id} | Get one outbound action. Elder sees elder_safe_message + status only |
| GET | /actions/outreach/summary | Counts (sent, suppressed, failed, queued, delivered) + recent list. Caregiver/admin only |

---

## Realtime

- **WebSocket** `WS /ws/risk_signals`: connect to receive new risk_signal payloads as they are created. Message shape: `{ "type": "risk_signal", "id": "<uuid>", "household_id": "<uuid>", "ts": "<iso>", "signal_type": "...", "severity": 1-5, "score": float }`.

---

## JSON schemas for UI

### Risk signal card (list item)

```json
{
  "id": "uuid",
  "ts": "ISO8601",
  "signal_type": "string",
  "severity": 1,
  "score": 0.0,
  "status": "open | acknowledged | dismissed | escalated",
  "summary": "optional string"
}
```

### Risk signal detail (with subgraph)

- **explanation**: `{ "summary": "...", "motif_tags": [...], "timeline_snippet": [...], "model_available": boolean, "subgraph" | "model_subgraph"?: { "nodes", "edges" } (may be absent when model_available is false), "top_entities": [...], "top_edges": [...], "session_ids": [...], "event_ids": [...], "entity_ids": [...] }`
- **recommended_action**: `{ "checklist": ["...", ...], "motif_context": [...], "severity": N, "escalation_draft"?: "..." }` (Financial Security Agent sets checklist array.)
- **subgraph** (for viz):

```json
{
  "nodes": [
    { "id": "string", "type": "string", "label": "string | null", "score": "number | null" }
  ],
  "edges": [
    { "src": "string", "dst": "string", "type": "string", "weight": "number | null", "rank": "number | null" }
  ]
}
```

### Weekly summary

```json
{
  "id": "uuid",
  "period_start": "ISO8601 | null",
  "period_end": "ISO8601 | null",
  "summary_text": "string",
  "summary_json": {}
}
```

### Watchlist item

```json
{
  "id": "uuid",
  "watch_type": "string",
  "pattern": {},
  "reason": "string | null",
  "priority": 0,
  "expires_at": "ISO8601 | null"
}
```

---

## UI inputs/outputs summary

| UI need | Endpoint / source | Shape |
|--------|-------------------|--------|
| Household + role | GET /households/me | `{ id, name, role, display_name }` |
| Session list (date range) | GET /sessions?from=&to= | `{ sessions: [...], total }` |
| Events for a session | GET /sessions/{id}/events | `{ events: [...], total, next_offset }` |
| Risk alerts list | GET /risk_signals | `{ signals: [...], total }` |
| Risk detail + graph viz | GET /risk_signals/{id} | Full + `explanation_json` (includes `model_available`; `model_subgraph` may be absent), `subgraph` |
| Similar incidents | GET /risk_signals/{id}/similar | `{ available: boolean, reason?: string, similar: [...] }`. When model did not run or no embedding: `available: false`, `reason: "model_not_run"`, `similar: []`. When available: each item has `risk_signal_id`, `similarity`, `score`, `ts`, `signal_type`, `severity`, `status`, `label_outcome`. |
| Caregiver feedback | POST /risk_signals/{id}/feedback | `{ label, notes }` |
| Device watchlists | GET /watchlists or POST /device/sync | `{ watchlists: [...] }` |
| Realtime risk push | WS /ws/risk_signals | JSON risk_signal object |
| Weekly rollup | GET /summaries | `[{ period_start, period_end, summary_text, ... }]` |
| Run Financial Agent | POST /agents/financial/run | `{ time_window_days?, dry_run? }` → run_id, counts, logs; dry_run returns risk_signals/watchlists |
| Run other agents | POST /agents/{drift,narrative,ring,calibration,redteam}/run | `{ dry_run? }` → run_id, step_trace, summary_json |
| Agents status | GET /agents/status | `{ agents: [{ agent_name, last_run_at, last_run_status, last_run_summary }] }` (includes last_run_summary for all) |
| Agent trace | GET /agents/trace?run_id=&agent_name= or GET /agents/{slug}/trace?run_id= | Single agent_runs row: step_trace, summary_json |
| Financial run trace | GET /agents/financial/trace?run_id= | Single agent_runs row for run_id |
| Notify caregiver | POST /actions/outreach | Body: risk_signal_id, channel_preference?, dry_run?. Returns outbound_action, agent_run_id, preview (if dry_run), suppressed, sent |
| Outreach list | GET /actions/outreach | { actions: [{ id, status, created_at, sent_at, error?, payload.elder_safe_message (elder only), ... }] } |
| Outreach summary | GET /actions/outreach/summary | { counts: { sent, suppressed, failed, queued, delivered }, recent: [{ id, status, created_at, sent_at, error, channel, recipient_contact_last4 }] } (caregiver/admin only) |

**Agent artifacts (summary_json.artifact_refs):** risk_signal_ids, watchlist_ids, ring_ids, summary_ids, replay_fixture_path where applicable. UI: /agents page shows Run, Dry run, View trace per agent; drift chart (Recharts); narrative "Evidence-only" badge on alert detail; ring_candidate "View ring" on alerts; calibration report; redteam pass rate and last failures.

---

## Agent summary_json shapes (for UI Results panel)

All agents persist `agent_runs` with `summary_json` that the UI can render. Each summary must include at least: **headline** (or title), **key_metrics**, **key_findings**, **recommended_actions**, **artifact_refs** (or artifact_refs embedded in summary_json).

### Graph Drift (`graph_drift`)

- **headline**: string (e.g. "Drift detected; review prototypes" or "No significant drift")
- **reason**: optional, e.g. "no_supabase", "insufficient_samples"
- **key_metrics**: `{ centroid_shift?, mmd?, pca_ks_aggregate?, norm_drift?, neighbor_stability_delta?, n_baseline?, n_recent? }`
- **quality_report**: `{ dim_consistent, dim, n_embeddings, has_nan, l2_norm_mean, l2_norm_stdev }`
- **top_slices**: list of `{ slice, baseline_count, recent_count, drift_delta, example_risk_signal_ids }`
- **prototypes**: list of `{ risk_signal_id, distance_from_baseline, summary }`
- **cause**: string (e.g. "model_change", "behavior_shift", "new_pattern")
- **key_findings**: string[]
- **recommended_actions**: string[]
- **artifact_refs**: `{ risk_signal_ids[], summary_ids[] }`

### Evidence Narrative / Investigation Packager (`evidence_narrative`)

- **headline**: string
- **key_metrics**: `{ signals_processed, updated }`
- **key_findings**: string[]
- **recommended_actions**: string[]
- **narrative_preview**: string (truncated)
- **hypotheses_sample**: list (optional)
- **elder_safe_sample**: `{ plain_language_summary?, do_now_checklist?, reassurance_line? }` (optional)
- **artifact_refs**: `{ risk_signal_ids[], summary_ids[] }`

### Ring Discovery (`ring_discovery`)

- **headline**: string (e.g. "Found N ring(s); M alert(s) created")
- **key_metrics**: `{ rings_found, risk_signals_created }`
- **key_findings**: string[]
- **recommended_actions**: string[]
- **neo4j_available**: boolean
- **artifact_refs**: `{ ring_ids[], risk_signal_ids[], watchlist_ids[] }`

### Continual Calibration (`continual_calibration`)

- **headline**: string (e.g. "Calibration complete; system confidence improved")
- **key_metrics**: `{ before_ece, after_ece, feedback_count }`
- **key_findings**: string[]
- **recommended_actions**: string[]
- **policy_patch**: `{ new_thresholds, changed_rules, suggested_watchlist_threshold, notes }`
- **artifact_refs**: `{ summary_ids[], risk_signal_ids[] }`

### Synthetic Red-Team (`synthetic_redteam`)

- **headline**: string (e.g. "Red-team pass rate: X; N failing")
- **key_metrics**: `{ regression_pass_rate, scenarios_generated }`
- **key_findings**: string[]
- **recommended_actions**: string[]
- **scenarios_generated**: int
- **regression_pass_rate**: float
- **regression_passed**: boolean
- **model_available**: boolean
- **failing_cases**: list of `{ scenario_id, assertion?, expected?, got? }`
- **artifact_refs**: `{ scenario_ids[], replay_fixture_path?, risk_signal_ids[] }`

### Caregiver Outreach (`caregiver_outreach`)

- **headline**: string (e.g. "Caregiver Outreach complete")
- **outbound_action_id**: uuid | null
- **suppressed**: boolean (consent_allow_outbound_contact=false)
- **sent**: boolean (provider sent successfully)
- **provider_status**: "sent" | "failed"
- **idempotent**: boolean (when already sent for this risk_signal)
- **status_reason**: string | null

### Outbound action (API response)

- **id**, **household_id**, **triggered_by_risk_signal_id**, **action_type**, **channel**
- **recipient_name**, **recipient_contact** (caregiver/admin only; elder never)
- **recipient_contact_last4** (safe for display)
- **payload**: `{ caregiver_message?, elder_safe_message, evidence_refs?, recommended_actions? }`
- **status**: queued | sent | delivered | failed | suppressed
- **provider**, **provider_message_id**, **error**
- **created_at**, **sent_at**, **delivered_at**
