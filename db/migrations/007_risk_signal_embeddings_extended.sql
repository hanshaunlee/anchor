-- Extend risk_signal_embeddings for real GNN embeddings: dim, model_name, checkpoint_id, has_embedding, meta.
-- Enables: similar incidents only from real embeddings; embedding-centroid watchlists; model_not_run when no embedding.

ALTER TABLE risk_signal_embeddings
  ADD COLUMN IF NOT EXISTS dim INT,
  ADD COLUMN IF NOT EXISTS model_name TEXT,
  ADD COLUMN IF NOT EXISTS checkpoint_id TEXT,
  ADD COLUMN IF NOT EXISTS has_embedding BOOLEAN NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS meta JSONB DEFAULT '{}'::jsonb;

-- Backfill: existing rows have real embeddings (no placeholder rows were written when model didn't run).
UPDATE risk_signal_embeddings SET has_embedding = true WHERE has_embedding IS NULL;
UPDATE risk_signal_embeddings SET dim = jsonb_array_length(embedding::jsonb) WHERE dim IS NULL AND embedding IS NOT NULL AND jsonb_typeof(embedding) = 'array';
UPDATE risk_signal_embeddings SET model_name = 'hgt_baseline' WHERE model_name IS NULL;

-- Index for similar-incidents: same household, recent first (90-day window).
CREATE INDEX IF NOT EXISTS idx_risk_signal_embeddings_household_created
  ON risk_signal_embeddings(household_id, created_at DESC);

COMMENT ON COLUMN risk_signal_embeddings.has_embedding IS 'False when model did not run or embedding failed; /similar returns available=false.';
COMMENT ON COLUMN risk_signal_embeddings.dim IS 'Embedding dimension (for centroid watchlists and validation).';
