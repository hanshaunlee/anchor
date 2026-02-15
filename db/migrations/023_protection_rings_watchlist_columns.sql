-- Protection: rings + watchlist_items columns for A1/A2 (signature dedupe, display, lifecycle).
-- Rings: member_count, top_members_display, last_seen_at, delta_members_* (021 already added fingerprint, status, first_seen_at).
-- Watchlist_items (020): ensure canonical_key for API; first_seen_at/last_seen_at/evidence_count if missing.

-- Rings: add columns for summary display and delta tracking
ALTER TABLE rings
  ADD COLUMN IF NOT EXISTS member_count INT NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS top_members_display TEXT[] NOT NULL DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS delta_members_added INT NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS delta_members_removed INT NOT NULL DEFAULT 0;

-- Use updated_at as last_seen_at where last_seen_at is null
UPDATE rings SET last_seen_at = updated_at WHERE last_seen_at IS NULL AND updated_at IS NOT NULL;

COMMENT ON COLUMN rings.member_count IS 'Cached count of ring_members for list view';
COMMENT ON COLUMN rings.top_members_display IS 'Human-readable labels for top members (from entities)';
COMMENT ON COLUMN rings.last_seen_at IS 'Last time this ring was seen/updated';
COMMENT ON COLUMN rings.delta_members_added IS 'New members since previous run';
COMMENT ON COLUMN rings.delta_members_removed IS 'Members removed since previous run';

-- Watchlist_items: canonical_key for unique upsert (alias for fingerprint when present)
ALTER TABLE watchlist_items
  ADD COLUMN IF NOT EXISTS canonical_key TEXT,
  ADD COLUMN IF NOT EXISTS first_seen_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS evidence_count INT NOT NULL DEFAULT 0;

-- Backfill: canonical_key = fingerprint, first/last from created_at/updated_at, evidence_count from array length
-- array_length('{}', 1) is NULL in PostgreSQL, so COALESCE to 0 to satisfy NOT NULL
UPDATE watchlist_items
SET canonical_key = COALESCE(canonical_key, fingerprint),
    first_seen_at = COALESCE(first_seen_at, created_at),
    last_seen_at = COALESCE(last_seen_at, updated_at),
    evidence_count = COALESCE(array_length(evidence_signal_ids, 1), 0)
WHERE canonical_key IS NULL OR first_seen_at IS NULL OR last_seen_at IS NULL OR evidence_count = 0;

-- Unique index for upsert by (household_id, canonical_key) when canonical_key is set
CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_items_household_canonical_key
  ON watchlist_items (household_id, canonical_key) WHERE canonical_key IS NOT NULL AND status = 'active';
