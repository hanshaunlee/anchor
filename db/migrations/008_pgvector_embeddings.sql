-- Optional: pgvector for similar-incidents cosine NN. Requires CREATE EXTENSION vector (Supabase has it).
-- When extension is unavailable, application falls back to JSONB + Python cosine.

CREATE EXTENSION IF NOT EXISTS vector;

-- Fixed dimension 32 (HGT hidden_channels). Rows with dim != 32 keep using JSONB path.
ALTER TABLE risk_signal_embeddings
  ADD COLUMN IF NOT EXISTS embedding_vector vector(32);

-- Backfill from JSONB where dimension matches
UPDATE risk_signal_embeddings
SET embedding_vector = (
  SELECT array_agg((x::float) ORDER BY ord)
  FROM jsonb_array_elements_text(embedding) WITH ORDINALITY AS t(x, ord)
)
WHERE has_embedding = true
  AND (dim = 32 OR (dim IS NULL AND jsonb_array_length(embedding) = 32))
  AND embedding IS NOT NULL
  AND jsonb_typeof(embedding) = 'array';

-- IVFFLAT for cosine distance (lists = 100 for small/medium tables)
CREATE INDEX IF NOT EXISTS idx_risk_signal_embeddings_vector_cosine
  ON risk_signal_embeddings
  USING ivfflat (embedding_vector vector_cosine_ops)
  WITH (lists = 100);

-- RPC: nearest neighbors by cosine similarity (same household, time window, exclude self)
CREATE OR REPLACE FUNCTION similar_incidents_by_vector(
  p_risk_signal_id UUID,
  p_household_id UUID,
  p_top_k INT DEFAULT 5,
  p_since TIMESTAMPTZ DEFAULT (now() - interval '90 days')
)
RETURNS TABLE (risk_signal_id UUID, similarity FLOAT)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY
  SELECT e2.risk_signal_id, 1 - (e1.embedding_vector <=> e2.embedding_vector) AS similarity
  FROM risk_signal_embeddings e1
  CROSS JOIN LATERAL (
    SELECT e2.risk_signal_id, e2.embedding_vector
    FROM risk_signal_embeddings e2
    WHERE e2.household_id = p_household_id
      AND e2.risk_signal_id != p_risk_signal_id
      AND e2.embedding_vector IS NOT NULL
      AND e2.has_embedding = true
      AND e2.created_at >= p_since
    ORDER BY e2.embedding_vector <=> e1.embedding_vector
    LIMIT p_top_k
  ) e2
  WHERE e1.risk_signal_id = p_risk_signal_id
    AND e1.household_id = p_household_id
    AND e1.embedding_vector IS NOT NULL
    AND e1.has_embedding = true;
END;
$$;

COMMENT ON COLUMN risk_signal_embeddings.embedding_vector IS 'pgvector(32) for cosine NN; backfilled from embedding where dim=32.';
