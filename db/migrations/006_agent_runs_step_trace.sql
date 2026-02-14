-- Add compact step trace to agent_runs for UI and failure containment.
-- step_trace: list of { "step": string, "status": "ok" | "failed", "error"?: string }

ALTER TABLE agent_runs
  ADD COLUMN IF NOT EXISTS step_trace JSONB NOT NULL DEFAULT '[]';

COMMENT ON COLUMN agent_runs.step_trace IS 'Compact list of pipeline steps and status for this run (e.g. ingest, normalize, detect, recommend_watchlist, persist).';
