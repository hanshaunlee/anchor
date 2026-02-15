-- Risk signals: fingerprint for compound upsert (same risk = update and compound, not duplicate).
ALTER TABLE risk_signals ADD COLUMN IF NOT EXISTS fingerprint TEXT;

CREATE INDEX IF NOT EXISTS idx_risk_signals_household_fingerprint_open
  ON risk_signals (household_id, fingerprint) WHERE status = 'open' AND fingerprint IS NOT NULL;

COMMENT ON COLUMN risk_signals.fingerprint IS 'Stable hash for dedupe: same fingerprint => compound score and refresh updated_at';
