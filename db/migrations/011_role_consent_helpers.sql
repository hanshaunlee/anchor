-- Role system & consent helpers.
-- Role model: user roles elder | caregiver | admin (user_role enum in 001); device role = device (service principal).
-- Optional future: trusted_contact as caregiver subtype (not in enum yet).
-- sessions.consent_state remains JSONB; normalized keys at read time:
--   consent_share_with_caregiver, consent_share_text, consent_allow_outbound_contact, caregiver_contact_policy.
-- This migration adds DB helper auth.user_can_contact() for RLS and API.

-- user_can_contact(): true if current auth user is caregiver or admin AND (when we have session context)
-- consent_allow_outbound_contact is true. For RLS we only check role; consent is enforced in application.
CREATE OR REPLACE FUNCTION public.user_can_contact()
RETURNS BOOLEAN AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.users
    WHERE id = auth.uid()
    AND role IN ('caregiver', 'admin')
  )
$$ LANGUAGE sql STABLE SECURITY DEFINER;

COMMENT ON FUNCTION public.user_can_contact() IS 'True if current user role is caregiver or admin; used for outbound_actions RLS. Consent is enforced in app.';
