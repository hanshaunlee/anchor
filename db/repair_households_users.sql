-- =============================================================================
-- Anchor: create only households + users (and required enum) if missing.
-- Safe to run after a partial bootstrap or if you added the signup trigger
-- before running the full schema. Run in Supabase Dashboard → SQL Editor.
-- =============================================================================

-- Enum required by users.role (skip if exists)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
    CREATE TYPE user_role AS ENUM ('elder', 'caregiver', 'admin');
  END IF;
END $$;

-- Core tables the signup trigger needs (skip if exist)
CREATE TABLE IF NOT EXISTS households (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  role user_role NOT NULL,
  display_name TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- RLS and helpers (idempotent: replace function, alter table is safe)
ALTER TABLE households ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION public.user_household_id()
RETURNS UUID AS $$
  SELECT household_id FROM public.users WHERE id = auth.uid()
$$ LANGUAGE sql STABLE SECURITY DEFINER;

CREATE OR REPLACE FUNCTION public.user_role()
RETURNS user_role AS $$
  SELECT role FROM public.users WHERE id = auth.uid()
$$ LANGUAGE sql STABLE SECURITY DEFINER;

-- Policies (drop if exist then create, so re-run is safe)
DROP POLICY IF EXISTS households_select ON households;
DROP POLICY IF EXISTS households_update ON households;
CREATE POLICY households_select ON households FOR SELECT USING (id = public.user_household_id());
CREATE POLICY households_update ON households FOR UPDATE USING (id = public.user_household_id());

DROP POLICY IF EXISTS users_select ON users;
DROP POLICY IF EXISTS users_update_self ON users;
CREATE POLICY users_select ON users FOR SELECT USING (household_id = public.user_household_id());
CREATE POLICY users_update_self ON users FOR UPDATE USING (id = auth.uid());

-- Service role (used by API/trigger) bypasses RLS; anon needs these for reads after signin.
