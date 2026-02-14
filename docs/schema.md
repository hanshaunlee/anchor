# Anchor Supabase schema (handoff)

## Core tables

- **households** (id, name, created_at)
- **users** (id = auth.users.id, household_id, role: elder | caregiver | admin, display_name)
- **devices** (id, household_id, device_type, firmware_version, public_key, last_seen_at)
- **sessions** (id, household_id, device_id, started_at, ended_at, mode: offline | online, consent_state jsonb)
- **events** (id, session_id, device_id, ts, seq, event_type, payload jsonb, payload_version, text_redacted default true, ingested_at)

## Derived tables

- **utterances** (id, session_id, ts, speaker: elder | agent | unknown, text, text_hash, intent, confidence)
- **summaries** (id, household_id, session_id?, period_start?, period_end?, summary_text, summary_json)
- **entities** (id, household_id, entity_type: person | org | phone | email | account | merchant | device | location | topic, canonical, canonical_hash, meta)
- **mentions** (id, session_id, utterance_id?, event_id?, entity_id, ts, span, confidence)
- **relationships** (id, household_id, src_entity_id, dst_entity_id, rel_type, weight, first_seen_at, last_seen_at, evidence)
- **risk_signals** (id, household_id, ts, signal_type, severity 1–5, score, explanation jsonb, recommended_action jsonb, status: open | acknowledged | dismissed | escalated)
- **watchlists** (id, household_id, watch_type, pattern jsonb, reason, priority, created_at, expires_at?)
- **device_sync_state** (device_id pk, last_upload_ts, last_upload_seq_by_session jsonb, last_watchlist_pull_at)
- **feedback** (id, household_id, risk_signal_id, user_id, label: true_positive | false_positive | unsure, notes)
- **agent_runs** (id, household_id, agent_name, started_at, ended_at, status, summary_json, step_trace) — agent run traces (e.g. financial_security); step_trace is a compact list of pipeline steps and status per run
- **risk_signal_embeddings** (risk_signal_id, household_id, embedding jsonb, dim, model_name, checkpoint_id, has_embedding, meta) — for similar-incidents (cosine similarity) and embedding-centroid watchlists; `has_embedding=false` when model did not run (see migration 007)
- **household_calibration** (household_id, severity_threshold_adjust) — feedback-driven threshold adjustment
- **session_embeddings** (session_id, embedding jsonb) — optional session-level embeddings

## RLS

- Users see only rows where `household_id` matches their `users.household_id`.
- Devices can insert events for their device and select watchlists for their household.
