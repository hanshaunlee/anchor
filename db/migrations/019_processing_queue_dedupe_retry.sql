-- Processing queue: dedupe, atomic claim, retry policy, payload versioning.

-- New columns (add if not exists for idempotency)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'processing_queue' AND column_name = 'dedupe_key') THEN
    ALTER TABLE processing_queue ADD COLUMN dedupe_key TEXT;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'processing_queue' AND column_name = 'attempt_count') THEN
    ALTER TABLE processing_queue ADD COLUMN attempt_count INT NOT NULL DEFAULT 0;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'processing_queue' AND column_name = 'next_attempt_at') THEN
    ALTER TABLE processing_queue ADD COLUMN next_attempt_at TIMESTAMPTZ;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'processing_queue' AND column_name = 'last_error') THEN
    ALTER TABLE processing_queue ADD COLUMN last_error TEXT;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'processing_queue' AND column_name = 'payload_version') THEN
    ALTER TABLE processing_queue ADD COLUMN payload_version INT NOT NULL DEFAULT 1;
  END IF;
END $$;

-- Dedupe: one pending/running per (dedupe_key). NULL dedupe_key = no dedupe.
CREATE UNIQUE INDEX IF NOT EXISTS idx_processing_queue_dedupe_pending
  ON processing_queue(dedupe_key)
  WHERE status IN ('pending', 'running') AND dedupe_key IS NOT NULL;

COMMENT ON COLUMN processing_queue.dedupe_key IS 'Optional: household_id|job_type|window_bucket to prevent enqueue storms';
COMMENT ON COLUMN processing_queue.attempt_count IS 'Incremented on each claim; used for retry backoff';
COMMENT ON COLUMN processing_queue.next_attempt_at IS 'When to retry if failed (transient); NULL = immediate';
COMMENT ON COLUMN processing_queue.last_error IS 'Last error message (for retry/failed)';
COMMENT ON COLUMN processing_queue.payload_version IS 'Schema version of payload for future evolution';

-- Atomic claim: take one pending job, set running, return it. Skip jobs with next_attempt_at in future.
CREATE OR REPLACE FUNCTION public.rpc_claim_processing_queue_job()
RETURNS TABLE (
  id UUID,
  household_id UUID,
  job_type TEXT,
  payload JSONB,
  payload_version INT,
  attempt_count INT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  RETURN QUERY
  WITH claimed AS (
    UPDATE processing_queue q
    SET status = 'running',
        started_at = now(),
        attempt_count = q.attempt_count + 1
    FROM (
      SELECT q2.id
      FROM processing_queue q2
      WHERE q2.status = 'pending'
        AND (q2.next_attempt_at IS NULL OR q2.next_attempt_at <= now())
      ORDER BY q2.created_at
      LIMIT 1
      FOR UPDATE SKIP LOCKED
    ) sel
    WHERE q.id = sel.id
    RETURNING q.id, q.household_id, q.job_type, q.payload, q.payload_version, q.attempt_count
  )
  SELECT c.id, c.household_id, c.job_type, c.payload, c.payload_version, c.attempt_count FROM claimed c;
END;
$$;

COMMENT ON FUNCTION public.rpc_claim_processing_queue_job() IS 'Atomically claim one pending job; returns row or empty. Worker uses this to avoid double-processing.';
