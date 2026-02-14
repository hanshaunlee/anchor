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
- **explanation**: Backend stores as `explanation`; may contain `summary`, `top_entities`, `top_edges`, `motifs` (motif_tags), `session_ids`, `event_ids`, `entity_ids`, and/or `subgraph` / `model_subgraph`.
- **subgraph** (for graph viz):  
  `nodes: [{ id, type, label?, score? }]`,  
  `edges: [{ src, dst, type, weight?, rank? }]`.  
  (API doc uses `importance` in prose; schemas use `score` on nodes, `weight`/`rank` on edges.)
- **UI**: Timeline from evidence pointers; graph evidence panel (React Flow); motif tags; recommended actions; feedback panel.

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

## Pipeline Steps (for Agent UX)

From `api/graph_state.py` and `api/pipeline.py`:  
Ingest → Normalize → GraphUpdate → RiskScore → Explain → ConsentGate → WatchlistSynthesis → EscalationDraft → Persist.  
No trace API yet; frontend can use client-side trace model and map page actions to steps until API adds trace endpoints.
