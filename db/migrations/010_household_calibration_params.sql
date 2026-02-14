-- Calibration agent: store Platt / temperature params and last_calibrated_at.

ALTER TABLE household_calibration
  ADD COLUMN IF NOT EXISTS calibration_params JSONB DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS last_calibrated_at TIMESTAMPTZ;

COMMENT ON COLUMN household_calibration.calibration_params IS 'e.g. { "platt_a": float, "platt_b": float } or { "temperature": float }';
COMMENT ON COLUMN household_calibration.last_calibrated_at IS 'When calibration was last updated from feedback.';
