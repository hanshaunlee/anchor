-- 128-D embeddings for retrieval/similarity/rings (Phase 1). Keeps embedding_vector (32) for backward compatibility.
-- Run after 008_pgvector_embeddings.sql. Application prefers v2 when present.

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE risk_signal_embeddings
  ADD COLUMN IF NOT EXISTS embedding_vector_v2 vector(128);

-- Backfill from JSONB where dimension is 128
UPDATE risk_signal_embeddings
SET embedding_vector_v2 = (
  SELECT array_agg((x::float) ORDER BY ord)
  FROM jsonb_array_elements_text(embedding) WITH ORDINALITY AS t(x, ord)
)
WHERE has_embedding = true
  AND (dim = 128 OR (dim IS NULL AND jsonb_array_length(embedding) = 128))
  AND embedding IS NOT NULL
  AND jsonb_typeof(embedding) = 'array';

CREATE INDEX IF NOT EXISTS idx_risk_signal_embeddings_vector_v2_cosine
  ON risk_signal_embeddings
  USING ivfflat (embedding_vector_v2 vector_cosine_ops)
  WITH (lists = 100);

-- RPC: nearest neighbors by cosine similarity using 128-D column (same household, time window, exclude self)
CREATE OR REPLACE FUNCTION similar_incidents_by_vector_v2(
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
  SELECT e2.risk_signal_id, 1 - (e1.embedding_vector_v2 <=> e2.embedding_vector_v2) AS similarity
  FROM risk_signal_embeddings e1
  CROSS JOIN LATERAL (
    SELECT e2.risk_signal_id, e2.embedding_vector_v2
    FROM risk_signal_embeddings e2
    WHERE e2.household_id = p_household_id
      AND e2.risk_signal_id != p_risk_signal_id
      AND e2.embedding_vector_v2 IS NOT NULL
      AND e2.has_embedding = true
      AND e2.created_at >= p_since
    ORDER BY e2.embedding_vector_v2 <=> e1.embedding_vector_v2
    LIMIT p_top_k
  ) e2
  WHERE e1.risk_signal_id = p_risk_signal_id
    AND e1.household_id = p_household_id
    AND e1.embedding_vector_v2 IS NOT NULL
    AND e1.has_embedding = true;
END;
$$;

COMMENT ON COLUMN risk_signal_embeddings.embedding_vector_v2 IS 'pgvector(128) for cosine NN; preferred over embedding_vector for similarity/rings.';
