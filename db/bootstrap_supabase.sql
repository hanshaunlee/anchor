-- =============================================================================
-- Anchor: full Supabase schema bootstrap (run once on a new project)
-- Run this in Supabase Dashboard → SQL Editor. Order: 001 → 006.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 001_initial_schema.sql
-- -----------------------------------------------------------------------------
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

CREATE TABLE device_sync_state (
  device_id UUID PRIMARY KEY REFERENCES devices(id) ON DELETE CASCADE,
  last_upload_ts TIMESTAMPTZ,
  last_upload_seq_by_session JSONB DEFAULT '{}',
  last_watchlist_pull_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

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

CREATE TABLE IF NOT EXISTS session_embeddings (
  session_id UUID PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  embedding JSONB,
  model_version TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 002_rls.sql
-- -----------------------------------------------------------------------------
ALTER TABLE households ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE utterances ENABLE ROW LEVEL SECURITY;
ALTER TABLE summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE mentions ENABLE ROW LEVEL SECURITY;
ALTER TABLE relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlists ENABLE ROW LEVEL SECURITY;
ALTER TABLE device_sync_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_embeddings ENABLE ROW LEVEL SECURITY;

-- Helpers in public (Supabase does not allow CREATE in auth schema from SQL Editor)
CREATE OR REPLACE FUNCTION public.user_household_id()
RETURNS UUID AS $$
  SELECT household_id FROM public.users WHERE id = auth.uid()
$$ LANGUAGE sql STABLE SECURITY DEFINER;

CREATE OR REPLACE FUNCTION public.user_role()
RETURNS user_role AS $$
  SELECT role FROM public.users WHERE id = auth.uid()
$$ LANGUAGE sql STABLE SECURITY DEFINER;

CREATE POLICY households_select ON households
  FOR SELECT USING (id = public.user_household_id());
CREATE POLICY households_update ON households
  FOR UPDATE USING (id = public.user_household_id());

CREATE POLICY users_select ON users
  FOR SELECT USING (household_id = public.user_household_id());
CREATE POLICY users_update_self ON users
  FOR UPDATE USING (id = auth.uid());

CREATE POLICY devices_select ON devices
  FOR SELECT USING (household_id = public.user_household_id());

CREATE POLICY sessions_all ON sessions
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY events_select ON events
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM sessions s WHERE s.id = events.session_id AND s.household_id = public.user_household_id()
    )
  );

CREATE POLICY events_insert ON events
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM sessions s
      WHERE s.id = events.session_id
        AND s.household_id = public.user_household_id()
    )
  );

CREATE POLICY utterances_all ON utterances
  FOR ALL USING (
    EXISTS (SELECT 1 FROM sessions s WHERE s.id = utterances.session_id AND s.household_id = public.user_household_id())
  );

CREATE POLICY summaries_all ON summaries
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY entities_all ON entities
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY mentions_all ON mentions
  FOR ALL USING (
    EXISTS (SELECT 1 FROM sessions s WHERE s.id = mentions.session_id AND s.household_id = public.user_household_id())
  );

CREATE POLICY relationships_all ON relationships
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY risk_signals_all ON risk_signals
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY watchlists_select ON watchlists
  FOR SELECT USING (household_id = public.user_household_id());

CREATE POLICY device_sync_state_select ON device_sync_state
  FOR SELECT USING (
    EXISTS (SELECT 1 FROM devices d WHERE d.id = device_sync_state.device_id AND d.household_id = public.user_household_id())
  );
CREATE POLICY device_sync_state_update ON device_sync_state
  FOR ALL USING (
    EXISTS (SELECT 1 FROM devices d WHERE d.id = device_sync_state.device_id AND d.household_id = public.user_household_id())
  );

CREATE POLICY feedback_all ON feedback
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY session_embeddings_all ON session_embeddings
  FOR ALL USING (
    EXISTS (SELECT 1 FROM sessions s WHERE s.id = session_embeddings.session_id AND s.household_id = public.user_household_id())
  );

-- -----------------------------------------------------------------------------
-- 003_risk_signal_embeddings.sql
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS risk_signal_embeddings (
  risk_signal_id UUID PRIMARY KEY REFERENCES risk_signals(id) ON DELETE CASCADE,
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  embedding JSONB NOT NULL,
  model_version TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_risk_signal_embeddings_household ON risk_signal_embeddings(household_id);

CREATE TABLE IF NOT EXISTS household_calibration (
  household_id UUID PRIMARY KEY REFERENCES households(id) ON DELETE CASCADE,
  severity_threshold_adjust FLOAT NOT NULL DEFAULT 0,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 004_rls_embeddings_calibration.sql
-- -----------------------------------------------------------------------------
ALTER TABLE risk_signal_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE household_calibration ENABLE ROW LEVEL SECURITY;

CREATE POLICY risk_signal_embeddings_all ON risk_signal_embeddings FOR ALL USING (household_id = public.user_household_id());
CREATE POLICY household_calibration_all ON household_calibration FOR ALL USING (household_id = public.user_household_id());

-- -----------------------------------------------------------------------------
-- 005_agent_runs.sql
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  agent_name TEXT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at TIMESTAMPTZ,
  status TEXT NOT NULL DEFAULT 'running',
  summary_json JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_household_agent ON agent_runs(household_id, agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_runs_started ON agent_runs(started_at DESC);

ALTER TABLE agent_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY agent_runs_all ON agent_runs
  FOR ALL USING (household_id = public.user_household_id());

-- -----------------------------------------------------------------------------
-- 006_agent_runs_step_trace.sql
-- -----------------------------------------------------------------------------
ALTER TABLE agent_runs
  ADD COLUMN IF NOT EXISTS step_trace JSONB NOT NULL DEFAULT '[]';

COMMENT ON COLUMN agent_runs.step_trace IS 'Compact list of pipeline steps and status for this run (e.g. ingest, normalize, detect, recommend_watchlist, persist).';

-- -----------------------------------------------------------------------------
-- 007_risk_signal_embeddings_extended.sql
-- -----------------------------------------------------------------------------
ALTER TABLE risk_signal_embeddings
  ADD COLUMN IF NOT EXISTS dim INT,
  ADD COLUMN IF NOT EXISTS model_name TEXT,
  ADD COLUMN IF NOT EXISTS checkpoint_id TEXT,
  ADD COLUMN IF NOT EXISTS has_embedding BOOLEAN NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS meta JSONB DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_risk_signal_embeddings_household_created
  ON risk_signal_embeddings(household_id, created_at DESC);
