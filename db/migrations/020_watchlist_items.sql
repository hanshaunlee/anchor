-- Normalized watchlist_items: human display, dedupe by fingerprint, batching.
-- RLS: household-scoped.

CREATE TABLE IF NOT EXISTS watchlist_items (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  batch_id UUID NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'expired', 'superseded')),
  category TEXT NOT NULL CHECK (category IN ('contact', 'phrase', 'topic', 'device_policy', 'bank', 'other')),
  type TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT,
  value_normalized TEXT,
  display_label TEXT,
  display_value TEXT,
  explanation TEXT,
  priority INT NOT NULL DEFAULT 5,
  score FLOAT,
  source_agent TEXT,
  source_run_id UUID REFERENCES agent_runs(id) ON DELETE SET NULL,
  evidence_signal_ids UUID[] DEFAULT '{}',
  expires_at TIMESTAMPTZ,
  fingerprint TEXT NOT NULL
);

CREATE UNIQUE INDEX idx_watchlist_items_household_fingerprint_active
  ON watchlist_items (household_id, fingerprint) WHERE status = 'active';

CREATE INDEX idx_watchlist_items_household_status_category ON watchlist_items(household_id, status, category);
CREATE INDEX idx_watchlist_items_household_batch ON watchlist_items(household_id, batch_id);
CREATE INDEX idx_watchlist_items_household_fingerprint ON watchlist_items(household_id, fingerprint);

ALTER TABLE watchlist_items ENABLE ROW LEVEL SECURITY;

CREATE POLICY watchlist_items_household ON watchlist_items
  FOR ALL USING (household_id = public.user_household_id());
