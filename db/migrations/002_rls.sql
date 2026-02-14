-- RLS: users access only their household; devices insert events for their device_id, select watchlists for household

ALTER TABLE households ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE utterances ENABLE ROW LEVEL SECURITY;
ALTER TABLE summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE mentions ENABLE ROW LEVEL SECURITY;
ALTER TABLE relationships ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlists ENABLE ROW LEVEL SECURITY;
ALTER TABLE device_sync_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_embeddings ENABLE ROW LEVEL SECURITY;

-- Helpers in public (Supabase hosted does not allow CREATE in auth schema from SQL Editor)
CREATE OR REPLACE FUNCTION public.user_household_id()
RETURNS UUID AS $$
  SELECT household_id FROM public.users WHERE id = auth.uid()
$$ LANGUAGE sql STABLE SECURITY DEFINER;

CREATE OR REPLACE FUNCTION public.user_role()
RETURNS user_role AS $$
  SELECT role FROM public.users WHERE id = auth.uid()
$$ LANGUAGE sql STABLE SECURITY DEFINER;

-- Households: user can read/update own household
CREATE POLICY households_select ON households
  FOR SELECT USING (id = public.user_household_id());
CREATE POLICY households_update ON households
  FOR UPDATE USING (id = public.user_household_id());

-- Users: user can read users in same household
CREATE POLICY users_select ON users
  FOR SELECT USING (household_id = public.user_household_id());
CREATE POLICY users_update_self ON users
  FOR UPDATE USING (id = auth.uid());

-- Devices: user can read devices in household; service role can insert/update
CREATE POLICY devices_select ON devices
  FOR SELECT USING (household_id = public.user_household_id());

-- Sessions: household scope
CREATE POLICY sessions_all ON sessions
  FOR ALL USING (household_id = public.user_household_id());

-- Events: household scope (via session/device); device insert via service role or anon with device_id check
CREATE POLICY events_select ON events
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM sessions s WHERE s.id = events.session_id AND s.household_id = public.user_household_id()
    )
  );

-- Events insert: allow for device belonging to household (typically via service role or device JWT)
-- For device uploads we use service_role or a dedicated device policy: insert where device_id in (select id from devices where household_id = public.user_household_id())
-- With Supabase Auth, device may use anon key + JWT claim device_id; here we allow insert if session belongs to household
CREATE POLICY events_insert ON events
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM sessions s
      WHERE s.id = events.session_id
        AND s.household_id = public.user_household_id()
    )
  );

-- Utterances, summaries, entities, mentions, relationships: household scope
CREATE POLICY utterances_all ON utterances
  FOR ALL USING (
    EXISTS (SELECT 1 FROM sessions s WHERE s.id = utterances.session_id AND s.household_id = public.user_household_id())
  );

CREATE POLICY summaries_all ON summaries
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY entities_all ON entities
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY mentions_all ON mentions
  FOR ALL USING (
    EXISTS (SELECT 1 FROM sessions s WHERE s.id = mentions.session_id AND s.household_id = public.user_household_id())
  );

CREATE POLICY relationships_all ON relationships
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY risk_signals_all ON risk_signals
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY watchlists_select ON watchlists
  FOR SELECT USING (household_id = public.user_household_id());

-- device_sync_state: device can read/update own row (device_id = claim or service)
CREATE POLICY device_sync_state_select ON device_sync_state
  FOR SELECT USING (
    EXISTS (SELECT 1 FROM devices d WHERE d.id = device_sync_state.device_id AND d.household_id = public.user_household_id())
  );
CREATE POLICY device_sync_state_update ON device_sync_state
  FOR ALL USING (
    EXISTS (SELECT 1 FROM devices d WHERE d.id = device_sync_state.device_id AND d.household_id = public.user_household_id())
  );

CREATE POLICY feedback_all ON feedback
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY session_embeddings_all ON session_embeddings
  FOR ALL USING (
    EXISTS (SELECT 1 FROM sessions s WHERE s.id = session_embeddings.session_id AND s.household_id = public.user_household_id())
  );

-- Watchlists: device needs to read; ensure device_id is in same household (handled by SELECT above)
-- No insert/update from client for watchlists; backend only.
