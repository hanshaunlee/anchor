# Anchor Frontend – Data Objects & API Summary

Reference: `docs/api_ui_contracts.md`, `apps/api/api/schemas.py`, `apps/api/api/routers/*`.

---

## Auth

- **Sign-in**: Supabase Auth (email/password or OTP). JWT in `Authorization: Bearer <token>`.
- **Roles**: `elder` | `caregiver` | `admin`. All access is household-scoped (RLS).

---

## Data Objects Used by UI

### HouseholdMe

- **Source**: `GET /households/me`
- **Shape**: `{ id: UUID, name: string, role: "elder"|"caregiver"|"admin", display_name: string | null }`
- **Use**: Nav, role-based views, household context.

---

### Session list item

- **Source**: `GET /sessions?from=&to=` (query: `from`, `to` datetime; response paginated)
- **Response**: `{ sessions: SessionListItem[], total: number }`
- **SessionListItem**:  
  `id`, `device_id`, `started_at`, `ended_at`, `mode` ("offline"|"online"), `consent_state` (object), `summary_text` (optional).

---

### Session detail

- Session detail is **session list item +** `GET /sessions/{id}/events` for events.
- No dedicated “session by id” endpoint; use list and match by id, or derive from list item.

---

### Event list item

- **Source**: `GET /sessions/{session_id}/events?limit=&offset=`
- **Response**: `{ events: EventListItem[], total: number, next_offset: number | null }`
- **EventListItem**: `id`, `ts`, `seq`, `event_type`, `payload` (object), `text_redacted` (boolean).
- **UI**: Show "Redacted due to consent" when `text_redacted`; collapsible payload JSON viewer.

---

### RiskSignal card (list)

- **Source**: `GET /risk_signals?status=&severity>=&limit=&offset=`
  - Query: `status` (open|acknowledged|dismissed|escalated), `severity>=` (1–5), `limit`, `offset`.
- **Response**: `{ signals: RiskSignalCard[], total: number }`
- **RiskSignalCard**:  
  `id`, `ts`, `signal_type`, `severity` (1–5), `score`, `status`, `summary` (optional).
- **UI**: Card shows signal_type, severity, score, status chip, “top reasons” (from explanation summary or motif_tags), timestamp, CTA “Investigate”.

---

### RiskSignal detail (with subgraph)

- **Source**: `GET /risk_signals/{id}`
- **RiskSignalDetail**:  
  `id`, `household_id`, `ts`, `signal_type`, `severity`, `score`, `status`,  
  `explanation` (dict), `recommended_action` (dict),  
  `subgraph` (optional), `session_ids`, `event_ids`, `entity_ids`.
- **explanation**: Backend stores as `explanation`; may contain `summary`, `narrative`, `narrative_evidence_only` (boolean — set by Evidence Narrative Agent; **UI: show “Evidence-only” badge**), `top_entities`, `top_edges`, `motifs` (motif_tags), `session_ids`, `event_ids`, `entity_ids`, `subgraph` / `model_subgraph`, and for **ring_candidate** signals `ring_id` (**UI: “View ring” badge**).
- **signal_type**: When `ring_candidate` show “View ring”; when `drift_warning` show **“Drift warning”** badge (Graph Drift Agent).
- **subgraph** (for graph viz):  
  `nodes: [{ id, type, label?, score? }]`,  
  `edges: [{ src, dst, type, weight?, rank? }]`.  
  (API doc uses `importance` in prose; schemas use `score` on nodes, `weight`/`rank` on edges.)
- **recommended_action**: May contain `checklist` (array of strings) or `steps`; UI renders both. Financial Security Agent sets `checklist`.
- **UI**: Timeline from evidence pointers; graph evidence panel (React Flow); motif tags; recommended actions; feedback panel; Evidence-only / View ring / Drift warning badges when applicable.

---

### Similar incidents

- **Source**: `GET /risk_signals/{id}/similar?top_k=`
- **Response**: `{ similar: SimilarIncident[] }`
- **SimilarIncident**: `risk_signal_id`, `score` (similarity), `outcome` ("confirmed_scam"|"false_positive"|"open"|null), `ts` (optional).
- **UI**: “Similar incidents” panel; hide if endpoint returns empty or errors.

---

### Watchlist item

- **Source**: `GET /watchlists`
- **Response**: `{ watchlists: WatchlistItem[] }`
- **WatchlistItem**: `id`, `watch_type`, `pattern` (object), `reason` (string | null), `priority` (number), `expires_at` (ISO | null).
- **UI**: List with priority, reason, expiry; “Delta view” / “export to device” read-only unless API supports edit.

---

### Summary item (weekly)

- **Source**: `GET /summaries?from=&to=&session_id=&limit=`
- **Response**: `WeeklySummary[]` (array, not wrapped).
- **WeeklySummary**: `id`, `period_start`, `period_end`, `summary_text`, `summary_json` (object).
- **UI**: Weekly rollups, trend charts (counts of motifs, alerts, confirmations if in summary_json).

---

### Feedback (risk signal)

- **Request**: `POST /risk_signals/{id}/feedback`  
  Body: `{ label: "true_positive" | "false_positive" | "unsure", notes?: string }`
- **Response**: `{ ok: true }`
- **UI**: Buttons “Confirm scam” / “False alarm” / “Unsure”, notes textarea, submit.

---

### WebSocket (live risk signals)

- **Endpoint**: `WS /ws/risk_signals` (connect with same origin; auth if API expects token in query or subprotocol).
- **Message shape** (server → client):  
  `{ type: "risk_signal", id: string (uuid), household_id: string (uuid), ts: string (ISO), signal_type: string, severity: number (1–5), score: number }`
- **UI**: Subscribe in caregiver dashboard; merge into TanStack Query cache for `risk_signals` list; fallback to polling if WS fails.

---

## Query Parameter Notes

- **Risk signals list**: Min severity uses query key `severity>=` (alias) in FastAPI.
- **Sessions**: Date range `from` / `to` (alias for `from_ts` / `to_ts`).
- **Summaries**: `from` / `to` for period filter; optional `session_id`, `limit`.

---

## Agents API

- **GET /agents/status**: `{ agents: [{ agent_name, last_run_at, last_run_status, last_run_summary }] }`. **last_run_summary** (when present) can contain:
  - **Graph Drift:** `drift_detected` (boolean), `metrics` (e.g. `centroid_shift`, `mmd`, `ks`), `cause`, `examples` (risk_signal_ids).
  - **Ring Discovery:** `rings_found`, `risk_signals_created`, `artifact_refs.ring_ids`.
  - **Continual Calibration:** `feedback_count`, `adjustment_applied`, `calibration_params`, `before_ece` / `after_ece`.
  - **Synthetic Red-Team:** `scenarios_generated`, `regression_pass_rate`, `regression_passed`, `failing_cases`, `model_available`.
  UI: Agent Center shows these per-agent summary lines (drift metrics, rings count, red-team pass rate / failures).
- **POST /agents/financial/run**: Body `{ time_window_days?, dry_run?, use_demo_events? }`. Returns `run_id`, `logs`, `motif_tags`, `timeline_snippet`, `risk_signals_count`, `watchlists_count`; if `dry_run` or `use_demo_events` also `risk_signals`, `watchlists`; if `use_demo_events` also `input_events`.
- **POST /agents/{drift,narrative,ring,calibration,redteam}/run**: Body `{ dry_run? }`. Returns `ok`, `run_id`, `step_trace`, `summary_json`. UI: Run / Dry run buttons per agent; trace appears when `run_id` is set.
- **GET /agents/financial/demo**: No auth. Run agent on built-in demo events; returns `input_events`, `input_summary`, `output`. UI: Scenario Replay “Load from API (no auth)”.
- **GET /agents/trace?run_id=&agent_name=** or **GET /agents/{slug}/trace?run_id=**: Single run trace (id, started_at, ended_at, status, summary_json, step_trace). UI: show trace for the selected run.
- **GET /agents/financial/trace?run_id=**: Financial run trace only.

---

## Pipeline Steps (for Agent UX)

From `api/graph_state.py` and `api/pipeline.py`:  
Ingest → Normalize → GraphUpdate → RiskScore → Explain → ConsentGate → WatchlistSynthesis → EscalationDraft → Persist.  
Financial Security Agent: POST /agents/financial/run with dry_run and/or use_demo_events for preview; GET /agents/financial/demo for no-auth demo; GET /agents/financial/trace for run trace.

---

## Graph API

- **GET /graph/evidence**: Household evidence subgraph (nodes, edges) for Graph view. Requires auth.
- **POST /graph/sync-neo4j**: Mirror evidence graph to Neo4j. Requires auth.
- **GET /graph/neo4j-status**: `{ enabled, browser_url? }`. UI: Graph view “Sync to Neo4j” and “Open in Neo4j Browser”.

---

## Ingest API

- **POST /ingest/events**: Body `{ events: [ { session_id, device_id, ts, seq, event_type, payload, payload_version? } ] }`. Sessions/devices must belong to household. UI: Ingest events page.
