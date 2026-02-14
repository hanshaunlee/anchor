-- Safe display for contact info: last4 for UI; never expose raw to elder.
-- RLS: only users (elder/caregiver/admin) can SELECT via user_household_id(); device has no users row so gets no rows.

ALTER TABLE outbound_actions
  ADD COLUMN IF NOT EXISTS recipient_contact_last4 TEXT;

COMMENT ON COLUMN outbound_actions.recipient_contact_last4 IS 'Last 4 chars of phone/email for display; raw recipient_contact never exposed to elder UI.';

-- Backfill last4 from recipient_contact where possible (optional)
UPDATE outbound_actions
SET recipient_contact_last4 = RIGHT(recipient_contact, 4)
WHERE recipient_contact IS NOT NULL AND recipient_contact != '' AND (recipient_contact_last4 IS NULL OR recipient_contact_last4 = '');

-- caregiver_contacts.channels: store { sms: { number_hash?, last4 }, email: { email_hash?, last4 } } for display; raw only if necessary.
-- No schema change required; document in API: when returning to elder, only last4/names.

-- RLS: Device cannot read outbound_actions or caregiver_contacts because they use user_household_id()
-- which is NULL for device (no row in public.users). No additional policy needed.
