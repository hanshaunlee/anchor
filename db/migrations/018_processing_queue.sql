-- Lightweight processing queue: ingest → supervisor runs.
-- Worker polls; API or trigger enqueues. Avoids "data is in DB but nothing happened."

CREATE TABLE IF NOT EXISTS processing_queue (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  job_type TEXT NOT NULL DEFAULT 'run_supervisor_ingest',  -- run_supervisor_ingest | run_supervisor_new_alert | ...
  payload JSONB NOT NULL DEFAULT '{}',                     -- e.g. { "time_window_days": 7, "risk_signal_id": "..." }
  status TEXT NOT NULL DEFAULT 'pending',                  -- pending | running | completed | failed
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  error_text TEXT,
  CONSTRAINT processing_queue_status_check CHECK (status IN ('pending', 'running', 'completed', 'failed'))
);

CREATE INDEX idx_processing_queue_status_created
  ON processing_queue(status, created_at)
  WHERE status = 'pending';

CREATE INDEX idx_processing_queue_household
  ON processing_queue(household_id, created_at DESC);

ALTER TABLE processing_queue ENABLE ROW LEVEL SECURITY;

CREATE POLICY processing_queue_household ON processing_queue
  FOR ALL USING (household_id = public.user_household_id());

COMMENT ON TABLE processing_queue IS 'Queue for supervisor/ingest jobs; worker polls and runs run_supervisor_ingest_pipeline.';
