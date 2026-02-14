-- =============================================================================
-- Anchor: drop the signup trigger so Auth signup no longer runs it.
-- Run this in Supabase Dashboard → SQL Editor if signup returns 500
-- (trigger fails when households/users are missing or trigger has errors).
-- After this, signup succeeds and the frontend creates household + user
-- via POST /households/onboard (Option C). Ensure db/repair_households_users.sql
-- has been run so the API can create the rows.
-- =============================================================================

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

-- Optional: drop the function too (no longer used)
DROP FUNCTION IF EXISTS public.handle_new_user();
