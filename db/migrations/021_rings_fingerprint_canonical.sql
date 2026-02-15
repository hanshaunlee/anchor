-- Rings: fingerprint for dedupe, status, summary fields, first_seen_at, signals_count.
-- Canonical view: status = 'active'. Dedupe on persist by fingerprint or overlap.

ALTER TABLE rings
  ADD COLUMN IF NOT EXISTS fingerprint TEXT,
  ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active',
  ADD COLUMN IF NOT EXISTS summary_label TEXT,
  ADD COLUMN IF NOT EXISTS summary_text TEXT,
  ADD COLUMN IF NOT EXISTS canonical_ring_id UUID REFERENCES rings(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS first_seen_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS signals_count INT DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_rings_household_status ON rings(household_id, status);
CREATE INDEX IF NOT EXISTS idx_rings_fingerprint ON rings(household_id, fingerprint) WHERE fingerprint IS NOT NULL AND status = 'active';

COMMENT ON COLUMN rings.fingerprint IS 'sha256(sorted member entity_ids) for dedupe';
COMMENT ON COLUMN rings.status IS 'active | superseded';
COMMENT ON COLUMN rings.canonical_ring_id IS 'NULL = this row is canonical; set when superseded by another ring';
