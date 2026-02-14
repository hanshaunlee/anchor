# Event packet format (Edge → Supabase)

Edge device emits append-only events, stored locally then **batch uploaded weekly**.

## Per-event fields

| Field | Type | Description |
|-------|------|-------------|
| session_id | uuid | Session identifier |
| device_id | uuid | Device identifier |
| ts | timestamptz | Event timestamp |
| seq | int | Monotonic sequence number per session |
| event_type | text | e.g. `wake`, `partial_asr`, `final_asr`, `intent`, `tool_call`, `tool_result`, `tts`, `error` |
| payload_version | int | Version of payload schema |
| payload | jsonb | See below |

## Payload variants (by event_type)

- **final_asr**: `{ "text"?: string, "text_hash"?: string, "lang"?: string, "confidence"?: float, "speaker"?: { "speaker_id"?, "role"?: "elder"|"agent"|"unknown" } }`
- **intent**: `{ "name": string, "slots"?: Record<string, any>, "confidence"?: float }`
- **speaker**: `{ "speaker_id"?, "role"? }`
- **device_state**: `{ "online": bool, "battery"?, "wifi_ssid_hash"? }`
- **optional embeddings**: `{ "audio_embedding"?: float[], "text_embedding"?: float[] }`

Privacy default: `text` may be omitted; always store `text_hash` when available.

## Batch ingest

- **POST /ingest/events** body: `{ "events": [ { "session_id", "device_id", "ts", "seq", "event_type", "payload_version", "payload" }, ... ] }`
- Response: `{ "ingested": N, "session_ids": [...], "last_ts": "..." }`
