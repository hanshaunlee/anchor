-- Narrative reports: one row per Evidence Narrative agent run (persisted artifact for "View report").
CREATE TABLE IF NOT EXISTS narrative_reports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  agent_run_id UUID REFERENCES agent_runs(id) ON DELETE SET NULL,
  risk_signal_ids UUID[] NOT NULL DEFAULT '{}',
  report_json JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_narrative_reports_household ON narrative_reports(household_id);
CREATE INDEX IF NOT EXISTS idx_narrative_reports_agent_run ON narrative_reports(agent_run_id);
CREATE INDEX IF NOT EXISTS idx_narrative_reports_created ON narrative_reports(created_at DESC);

ALTER TABLE narrative_reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY narrative_reports_all ON narrative_reports
  FOR ALL USING (household_id = public.user_household_id());

COMMENT ON TABLE narrative_reports IS 'Evidence Narrative agent output: caregiver narrative, elder-safe, hypotheses per signal; referenced by artifact_refs.narrative_report_id';
