-- Performance indexes for compound endpoints and frequent queries.
-- risk_signal_embeddings: similar/drift; outbound_actions: candidates/history;
-- events: ingest window and evidence graph; risk_signals: list/detail.

-- risk_signal_embeddings: used by similar incidents and drift
CREATE INDEX IF NOT EXISTS idx_risk_signal_embeddings_household_created
  ON risk_signal_embeddings(household_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_signal_embeddings_risk_signal_id
  ON risk_signal_embeddings(risk_signal_id);

-- outbound_actions: used by alert page and outreach history
CREATE INDEX IF NOT EXISTS idx_outbound_actions_household_created
  ON outbound_actions(household_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_outbound_actions_signal_status
  ON outbound_actions(triggered_by_risk_signal_id, status)
  WHERE triggered_by_risk_signal_id IS NOT NULL;

-- events: ingest window and evidence; session_id + ts already in 001 (idx_events_session_ts)
-- Add household-scoped via sessions for window queries (events -> sessions.household_id)
CREATE INDEX IF NOT EXISTS idx_events_session_ts_desc
  ON events(session_id, ts DESC);

-- risk_signals: list and filters; "open alerts" is a constant query pattern
CREATE INDEX IF NOT EXISTS idx_risk_signals_household_ts_desc
  ON risk_signals(household_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_risk_signals_household_status_ts
  ON risk_signals(household_id, status, ts DESC);
CREATE INDEX IF NOT EXISTS idx_risk_signals_status_severity
  ON risk_signals(status, severity DESC);

-- outbound_actions: candidates / queued often filtered by status
CREATE INDEX IF NOT EXISTS idx_outbound_actions_queued
  ON outbound_actions(household_id, created_at DESC)
  WHERE status = 'queued';
