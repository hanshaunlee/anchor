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
| GET | /sessions?from=&to= | List sessions (id, device_id, timestamps, consent_state, summary) |
| GET | /sessions/{id}/events | Paginated events (redacted by consent) |
| GET | /risk_signals?status=&severity>= | List risk signals |
| GET | /risk_signals/{id} | Full detail + explanation_json + evidence pointers |
| POST | /risk_signals/{id}/feedback | Submit label (true_positive / false_positive / unsure) + notes |
| GET | /watchlists | Watchlists for device + UI |
| POST | /device/sync | Device heartbeat; returns watchlists delta + last upload pointers |
| POST | /ingest/events | Batch ingest events (device authenticated) |
| GET | /summaries?from=&to= | Weekly rollups / session summaries |

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

- **explanation_json**: `{ "summary": "...", "top_entities": [...], "top_edges": [...], "motifs": [...], "session_ids": [...], "event_ids": [...], "entity_ids": [...] }`
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
| Risk detail + graph viz | GET /risk_signals/{id} | Full + `explanation_json`, `subgraph` |
| Caregiver feedback | POST /risk_signals/{id}/feedback | `{ label, notes }` |
| Device watchlists | GET /watchlists or POST /device/sync | `{ watchlists: [...] }` |
| Realtime risk push | WS /ws/risk_signals | JSON risk_signal object |
| Weekly rollup | GET /summaries | `[{ period_start, period_end, summary_text, ... }]` |
