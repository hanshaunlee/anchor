# Event packet format (Edge → Supabase)

Edge device emits append-only events, stored locally then **batch uploaded weekly**.

**Contract of record:** The canonical implementation is the Pydantic schema in `api.schemas` (`EventPacket` and payload variants). This doc is the human-readable spec; code enforces strict validation, versioning, and rejection logging.

## Per-event fields

| Field | Type | Description |
|-------|------|-------------|
| session_id | uuid | Session identifier |
| device_id | uuid | Device identifier |
| ts | timestamptz | Event timestamp |
| seq | int | Monotonic sequence number per session |
| event_type | text | e.g. `wake`, `partial_asr`, `final_asr`, `intent`, `tool_call`, `tool_result`, `tts`, `error`; financial: `transaction_detected`, `payee_added`, `bank_alert_received` |
| payload_version | int | Version of payload schema |
| payload | jsonb | See below |

## Payload variants (by event_type)

- **final_asr**: `{ "text"?: string, "text_hash"?: string, "lang"?: string, "confidence"?: float, "speaker"?: { "speaker_id"?, "role"?: "elder"|"agent"|"unknown" } }`
- **intent**: `{ "name": string, "slots"?: Record<string, any>, "confidence"?: float }`
- **speaker**: `{ "speaker_id"?, "role"? }`
- **device_state**: `{ "online": bool, "battery"?, "wifi_ssid_hash"? }`
- **optional embeddings**: `{ "audio_embedding"?: float[], "text_embedding"?: float[] }`
- **Financial (same pipeline as above; no separate transactions/accounts schema):**
  - **transaction_detected**: `{ "merchant"?: string, "amount_currency"?: string, "account_id_hash"?: string, "confidence"?: float }`
  - **payee_added**: `{ "payee_name"?: string, "payee_type"?: "person"|"merchant", "confidence"?: float }`
  - **bank_alert_received**: `{ "alert_type"?: string, "account_id_hash"?: string, "confidence"?: float }`

These financial event types are stored as events and normalized into entities/relationships the same way (e.g. merchant, person, account from payload → entities and mentions). Optional demo adapters (e.g. Plaid Sandbox) can emit the same event packets.

Privacy default: `text` may be omitted; always store `text_hash` when available.

## Batch ingest

- **POST /ingest/events** body: `{ "events": [ { "session_id", "device_id", "ts", "seq", "event_type", "payload_version", "payload" }, ... ] }`
- Response: `{ "ingested": N, "session_ids": [...], "last_ts": "...", "rejected": M, "rejection_reasons": [...] }`
- Invalid events are rejected and logged; only valid events are stored. Supported `payload_version` values are enforced (see schema).

## Failure containment

- **No graph mutation if ASR/intent confidence is low.** Events are always ingested to Supabase when valid; the derived graph (utterances, entities, mentions) is only updated when confidence meets the configured minimum (`config.settings.PipelineSettings.asr_confidence_min_for_graph`). Policy: `config.graph_policy.allow_graph_mutation_for_event`.
