-- Anchor: initial schema (Supabase / Postgres)
-- Core + derived tables; RLS in 002_rls.sql

-- Enums
CREATE TYPE user_role AS ENUM ('elder', 'caregiver', 'admin');
CREATE TYPE session_mode AS ENUM ('offline', 'online');
CREATE TYPE speaker_type AS ENUM ('elder', 'agent', 'unknown');
CREATE TYPE entity_type_enum AS ENUM (
  'person', 'org', 'phone', 'email', 'account', 'merchant', 'device', 'location', 'topic'
);
CREATE TYPE risk_signal_status AS ENUM ('open', 'acknowledged', 'dismissed', 'escalated');
CREATE TYPE feedback_label AS ENUM ('true_positive', 'false_positive', 'unsure');

-- Core tables
CREATE TABLE households (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE users (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  role user_role NOT NULL,
  display_name TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE devices (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  device_type TEXT,
  firmware_version TEXT,
  public_key TEXT,
  last_seen_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  device_id UUID NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  started_at TIMESTAMPTZ NOT NULL,
  ended_at TIMESTAMPTZ,
  mode session_mode NOT NULL DEFAULT 'offline',
  consent_state JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  device_id UUID NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  ts TIMESTAMPTZ NOT NULL,
  seq INT NOT NULL,
  event_type TEXT NOT NULL,
  payload JSONB NOT NULL DEFAULT '{}',
  payload_version INT NOT NULL DEFAULT 1,
  text_redacted BOOLEAN NOT NULL DEFAULT true,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (session_id, seq)
);

CREATE INDEX idx_events_session_ts ON events(session_id, ts);
CREATE INDEX idx_events_household ON events(device_id, ts);
CREATE INDEX idx_events_ingested ON events(ingested_at);

-- Derived: utterances (from final_asr / intent events)
CREATE TABLE utterances (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  ts TIMESTAMPTZ NOT NULL,
  speaker speaker_type NOT NULL DEFAULT 'unknown',
  text TEXT,
  text_hash TEXT,
  intent TEXT,
  confidence FLOAT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_utterances_session ON utterances(session_id, ts);

-- Derived: summaries (weekly rollups, etc.)
CREATE TABLE summaries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
  period_start TIMESTAMPTZ,
  period_end TIMESTAMPTZ,
  summary_text TEXT,
  summary_json JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_summaries_household_period ON summaries(household_id, period_start, period_end);

-- Derived: entities (canonicalized)
CREATE TABLE entities (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  entity_type entity_type_enum NOT NULL,
  canonical TEXT NOT NULL,
  canonical_hash TEXT,
  meta JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_entities_household_type_hash ON entities(household_id, entity_type, COALESCE(canonical_hash, canonical));
CREATE INDEX idx_entities_household ON entities(household_id);

-- Derived: mentions (entity in session/utterance/event)
CREATE TABLE mentions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  utterance_id UUID REFERENCES utterances(id) ON DELETE SET NULL,
  event_id UUID REFERENCES events(id) ON DELETE SET NULL,
  entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  ts TIMESTAMPTZ NOT NULL,
  span JSONB,
  confidence FLOAT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_mentions_entity ON mentions(entity_id, ts);
CREATE INDEX idx_mentions_session ON mentions(session_id, ts);

-- Derived: relationships (entity-entity with temporal attributes)
CREATE TABLE relationships (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  src_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  dst_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  rel_type TEXT NOT NULL,
  weight FLOAT NOT NULL DEFAULT 1.0,
  first_seen_at TIMESTAMPTZ NOT NULL,
  last_seen_at TIMESTAMPTZ NOT NULL,
  evidence JSONB DEFAULT '[]',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (household_id, src_entity_id, dst_entity_id, rel_type)
);

CREATE INDEX idx_relationships_household ON relationships(household_id);
CREATE INDEX idx_relationships_src_dst ON relationships(src_entity_id, dst_entity_id);

-- Risk signals (output of ML pipeline)
CREATE TABLE risk_signals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  signal_type TEXT NOT NULL,
  severity INT NOT NULL CHECK (severity >= 1 AND severity <= 5),
  score FLOAT NOT NULL,
  explanation JSONB DEFAULT '{}',
  recommended_action JSONB DEFAULT '{}',
  status risk_signal_status NOT NULL DEFAULT 'open',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_risk_signals_household_status ON risk_signals(household_id, status);
CREATE INDEX idx_risk_signals_ts ON risk_signals(ts);

-- Watchlists (pushed to device)
CREATE TABLE watchlists (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  watch_type TEXT NOT NULL,
  pattern JSONB NOT NULL DEFAULT '{}',
  reason TEXT,
  priority INT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ
);

CREATE INDEX idx_watchlists_household ON watchlists(household_id);

-- Device sync state (last upload tracking)
CREATE TABLE device_sync_state (
  device_id UUID PRIMARY KEY REFERENCES devices(id) ON DELETE CASCADE,
  last_upload_ts TIMESTAMPTZ,
  last_upload_seq_by_session JSONB DEFAULT '{}',
  last_watchlist_pull_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Feedback (caregiver labels on risk signals)
CREATE TABLE feedback (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  risk_signal_id UUID NOT NULL REFERENCES risk_signals(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  label feedback_label NOT NULL,
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_feedback_risk_signal ON feedback(risk_signal_id);

-- Optional: embeddings store (pgvector if extension enabled; else JSONB)
-- CREATE EXTENSION IF NOT EXISTS vector;
-- CREATE TABLE entity_embeddings (entity_id UUID PRIMARY KEY REFERENCES entities(id), embedding vector(256));
-- For now use JSONB in meta or separate table without vector
CREATE TABLE IF NOT EXISTS session_embeddings (
  session_id UUID PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  embedding JSONB,
  model_version TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
