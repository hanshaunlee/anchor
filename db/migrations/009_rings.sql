-- Ring discovery agent: store ring candidates and membership.
-- RLS: household-scoped.

CREATE TABLE IF NOT EXISTS rings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  score FLOAT NOT NULL DEFAULT 0,
  meta JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ring_members (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ring_id UUID NOT NULL REFERENCES rings(id) ON DELETE CASCADE,
  entity_id UUID REFERENCES entities(id) ON DELETE SET NULL,
  role TEXT,
  first_seen_at TIMESTAMPTZ,
  last_seen_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (ring_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_rings_household ON rings(household_id);
CREATE INDEX IF NOT EXISTS idx_ring_members_ring ON ring_members(ring_id);

ALTER TABLE rings ENABLE ROW LEVEL SECURITY;
ALTER TABLE ring_members ENABLE ROW LEVEL SECURITY;

CREATE POLICY rings_all ON rings FOR ALL USING (household_id = public.user_household_id());
CREATE POLICY ring_members_all ON ring_members FOR ALL USING (
  EXISTS (SELECT 1 FROM rings r WHERE r.id = ring_members.ring_id AND r.household_id = public.user_household_id())
);
