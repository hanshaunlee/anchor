-- Risk signal embeddings for Similar Incidents (cosine retrieval)
CREATE TABLE IF NOT EXISTS risk_signal_embeddings (
  risk_signal_id UUID PRIMARY KEY REFERENCES risk_signals(id) ON DELETE CASCADE,
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  embedding JSONB NOT NULL,
  model_version TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_risk_signal_embeddings_household ON risk_signal_embeddings(household_id);

-- Per-household threshold calibration (HITL feedback)
CREATE TABLE IF NOT EXISTS household_calibration (
  household_id UUID PRIMARY KEY REFERENCES households(id) ON DELETE CASCADE,
  severity_threshold_adjust FLOAT NOT NULL DEFAULT 0,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
