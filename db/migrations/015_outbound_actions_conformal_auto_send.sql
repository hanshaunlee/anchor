-- Outbound actions: conformal/calibration context at send; draft status.
-- Household capabilities: auto_send_outreach for gated auto-send.

-- Add conformal and score context to outbound_actions (nullable for existing rows)
ALTER TABLE outbound_actions
  ADD COLUMN IF NOT EXISTS conformal_triggered BOOLEAN DEFAULT false,
  ADD COLUMN IF NOT EXISTS calibrated_p_at_send DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS fusion_score_at_send DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS decision_rule_used TEXT;

-- Optional: add 'draft' to status enum (run once; skip if already applied)
-- ALTER TYPE outbound_action_status ADD VALUE IF NOT EXISTS 'draft';  -- PG 15+
-- Until then: use status 'queued' with sent_at NULL for draft-like rows.

-- Household capabilities: allow auto-send of outreach when consent + severity + conformal allow
ALTER TABLE household_capabilities
  ADD COLUMN IF NOT EXISTS auto_send_outreach BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN outbound_actions.conformal_triggered IS 'True when escalation was triggered by conformal rule (1 - calibrated_p >= q_hat)';
COMMENT ON COLUMN outbound_actions.calibrated_p_at_send IS 'Platt-calibrated probability at time of send';
COMMENT ON COLUMN outbound_actions.decision_rule_used IS 'raw_threshold | calibrated | conformal | rule_only';
COMMENT ON COLUMN household_capabilities.auto_send_outreach IS 'When true, supervisor NEW_ALERT may auto-send outreach if consent and conformal allow';
