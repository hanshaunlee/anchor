-- One-time setup: enable Notify caregiver & Action plan for household d38bd799-fc35-44f4-bcab-e9ff5ad64c35
-- Run this in Supabase Dashboard → SQL Editor. Copy all and click Run.

-- 1) Household consent: allow outbound contact
INSERT INTO household_consent_defaults (household_id, share_with_caregiver, share_text, allow_outbound_contact, escalation_threshold, updated_at)
VALUES (
  'd38bd799-fc35-44f4-bcab-e9ff5ad64c35'::uuid,
  true,
  true,
  true,
  3,
  now()
)
ON CONFLICT (household_id) DO UPDATE SET
  allow_outbound_contact = true,
  updated_at = now();

-- 2) Latest session: set consent so agents see it (no-op if no sessions yet)
UPDATE sessions
SET consent_state = COALESCE(consent_state, '{}'::jsonb) || '{"consent_allow_outbound_contact": true}'::jsonb
WHERE household_id = 'd38bd799-fc35-44f4-bcab-e9ff5ad64c35'::uuid
  AND id = (
    SELECT id FROM sessions
    WHERE household_id = 'd38bd799-fc35-44f4-bcab-e9ff5ad64c35'::uuid
    ORDER BY started_at DESC
    LIMIT 1
  );

-- 3) One caregiver contact (only if none exist for this household)
INSERT INTO caregiver_contacts (household_id, user_id, name, relationship, channels, priority, quiet_hours, verified, created_at, updated_at)
SELECT
  'd38bd799-fc35-44f4-bcab-e9ff5ad64c35'::uuid,
  'b1822f1d-7ad5-47ff-bb65-8168003dce76'::uuid,
  'Demo Caregiver',
  'family',
  '{"email": {"email": "demo@example.com", "value": "demo@example.com"}}'::jsonb,
  1,
  '{}'::jsonb,
  false,
  now(),
  now()
FROM (SELECT 1) AS _dummy
WHERE NOT EXISTS (
  SELECT 1 FROM caregiver_contacts WHERE household_id = 'd38bd799-fc35-44f4-bcab-e9ff5ad64c35'::uuid LIMIT 1
);
