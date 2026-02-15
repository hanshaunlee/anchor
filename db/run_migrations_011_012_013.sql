-- =============================================================================
-- Run this in Supabase Dashboard → SQL Editor (paste all, then Run).
-- Creates tables required for Notify caregiver & Action plan:
--   user_can_contact(), outbound_actions, caregiver_contacts,
--   household_capabilities, household_consent_defaults, action_playbooks,
--   action_tasks, incident_packets.
-- If you see "already exists" for a type/table, that migration was run before;
-- you can re-run this script — CREATE OR REPLACE and ADD COLUMN IF NOT EXISTS
-- are safe; duplicate CREATE TYPE/CREATE TABLE will error (run the rest manually
-- or ignore the first failure and run from the next statement).
-- =============================================================================

-- ---------- 011_role_consent_helpers ----------
CREATE OR REPLACE FUNCTION public.user_can_contact()
RETURNS BOOLEAN AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.users
    WHERE id = auth.uid()
    AND role IN ('caregiver', 'admin')
  )
$$ LANGUAGE sql STABLE SECURITY DEFINER;

COMMENT ON FUNCTION public.user_can_contact() IS 'True if current user role is caregiver or admin; used for outbound_actions RLS.';

-- ---------- 012_outbound_actions_caregiver_contacts ----------
DO $$ BEGIN
  CREATE TYPE outbound_action_type AS ENUM ('caregiver_notify', 'caregiver_call', 'caregiver_email');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$ BEGIN
  CREATE TYPE outbound_channel AS ENUM ('sms', 'email', 'voice_call');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$ BEGIN
  CREATE TYPE outbound_action_status AS ENUM ('queued', 'sent', 'delivered', 'failed', 'suppressed');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$ BEGIN
  CREATE TYPE outbound_provider AS ENUM ('twilio', 'sendgrid', 'smtp', 'mock');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS outbound_actions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  triggered_by_risk_signal_id UUID REFERENCES risk_signals(id) ON DELETE SET NULL,
  triggered_by_agent_run_id UUID REFERENCES agent_runs(id) ON DELETE SET NULL,
  action_type outbound_action_type NOT NULL,
  channel outbound_channel NOT NULL,
  recipient_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  recipient_name TEXT,
  recipient_contact TEXT,
  payload JSONB NOT NULL DEFAULT '{}',
  status outbound_action_status NOT NULL DEFAULT 'queued',
  provider outbound_provider NOT NULL DEFAULT 'mock',
  provider_message_id TEXT,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  sent_at TIMESTAMPTZ,
  delivered_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_outbound_actions_household ON outbound_actions(household_id);
CREATE INDEX IF NOT EXISTS idx_outbound_actions_risk_signal ON outbound_actions(triggered_by_risk_signal_id);
CREATE INDEX IF NOT EXISTS idx_outbound_actions_created ON outbound_actions(created_at DESC);

CREATE TABLE IF NOT EXISTS caregiver_contacts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  name TEXT NOT NULL,
  relationship TEXT,
  channels JSONB NOT NULL DEFAULT '{}',
  priority INT NOT NULL DEFAULT 1,
  quiet_hours JSONB DEFAULT '{}',
  verified BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_caregiver_contacts_household ON caregiver_contacts(household_id);

ALTER TABLE outbound_actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE caregiver_contacts ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS outbound_actions_select_household ON outbound_actions;
CREATE POLICY outbound_actions_select_household ON outbound_actions
  FOR SELECT USING (household_id = public.user_household_id());
DROP POLICY IF EXISTS outbound_actions_insert_caregiver_admin ON outbound_actions;
CREATE POLICY outbound_actions_insert_caregiver_admin ON outbound_actions
  FOR INSERT WITH CHECK (
    household_id = public.user_household_id() AND public.user_can_contact()
  );
DROP POLICY IF EXISTS outbound_actions_update_caregiver_admin ON outbound_actions;
CREATE POLICY outbound_actions_update_caregiver_admin ON outbound_actions
  FOR UPDATE USING (
    household_id = public.user_household_id() AND public.user_can_contact()
  );

DROP POLICY IF EXISTS caregiver_contacts_all_caregiver_admin ON caregiver_contacts;
CREATE POLICY caregiver_contacts_all_caregiver_admin ON caregiver_contacts
  FOR ALL USING (
    household_id = public.user_household_id() AND public.user_can_contact()
  );

-- recipient_contact_last4 for safe display (013_outbound_contact_safe_display)
ALTER TABLE outbound_actions ADD COLUMN IF NOT EXISTS recipient_contact_last4 TEXT;

-- ---------- 013_action_playbooks_capabilities_incident ----------
DO $$ BEGIN
  CREATE TYPE playbook_type AS ENUM ('bank_contact', 'account_lockdown', 'device_high_risk_mode', 'incident_response');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$ BEGIN
  CREATE TYPE playbook_status AS ENUM ('active', 'completed', 'canceled');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$ BEGIN
  CREATE TYPE action_task_type AS ENUM (
    'call_bank', 'email_bank', 'enable_alerts', 'freeze_card_instruction',
    'change_password_instruction', 'device_high_risk_mode_push',
    'verify_with_elder', 'notify_caregiver', 'file_report', 'lock_card'
  );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$ BEGIN
  CREATE TYPE action_task_status AS ENUM ('ready', 'blocked', 'done', 'skipped');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$ BEGIN
  CREATE TYPE bank_data_connector_enum AS ENUM ('none', 'plaid', 'open_banking', 'custom');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS household_capabilities (
  household_id UUID PRIMARY KEY REFERENCES households(id) ON DELETE CASCADE,
  notify_sms_enabled BOOLEAN NOT NULL DEFAULT false,
  notify_email_enabled BOOLEAN NOT NULL DEFAULT false,
  device_policy_push_enabled BOOLEAN NOT NULL DEFAULT true,
  bank_data_connector bank_data_connector_enum NOT NULL DEFAULT 'none',
  bank_control_capabilities JSONB NOT NULL DEFAULT '{"lock_card": false, "disable_transfers": false, "enable_alerts": true, "open_dispute": false}',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_household_capabilities_household ON household_capabilities(household_id);

CREATE TABLE IF NOT EXISTS household_consent_defaults (
  household_id UUID PRIMARY KEY REFERENCES households(id) ON DELETE CASCADE,
  share_with_caregiver BOOLEAN NOT NULL DEFAULT true,
  share_text BOOLEAN NOT NULL DEFAULT true,
  allow_outbound_contact BOOLEAN NOT NULL DEFAULT false,
  escalation_threshold INT NOT NULL DEFAULT 3,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT escalation_threshold_range CHECK (escalation_threshold >= 1 AND escalation_threshold <= 5)
);

CREATE INDEX IF NOT EXISTS idx_household_consent_defaults_household ON household_consent_defaults(household_id);

CREATE TABLE IF NOT EXISTS action_playbooks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  risk_signal_id UUID NOT NULL REFERENCES risk_signals(id) ON DELETE CASCADE,
  playbook_type playbook_type NOT NULL,
  graph JSONB NOT NULL DEFAULT '{"nodes": [], "edges": []}',
  status playbook_status NOT NULL DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_action_playbooks_household ON action_playbooks(household_id);
CREATE INDEX IF NOT EXISTS idx_action_playbooks_risk_signal ON action_playbooks(risk_signal_id);
CREATE INDEX IF NOT EXISTS idx_action_playbooks_status ON action_playbooks(status);

CREATE TABLE IF NOT EXISTS action_tasks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  playbook_id UUID NOT NULL REFERENCES action_playbooks(id) ON DELETE CASCADE,
  task_type action_task_type NOT NULL,
  status action_task_status NOT NULL DEFAULT 'ready',
  details JSONB NOT NULL DEFAULT '{}',
  completed_by_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_action_tasks_playbook ON action_tasks(playbook_id);
CREATE INDEX IF NOT EXISTS idx_action_tasks_status ON action_tasks(status);

CREATE TABLE IF NOT EXISTS incident_packets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  risk_signal_id UUID NOT NULL REFERENCES risk_signals(id) ON DELETE CASCADE,
  packet_json JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_incident_packets_household ON incident_packets(household_id);
CREATE INDEX IF NOT EXISTS idx_incident_packets_risk_signal ON incident_packets(risk_signal_id);

ALTER TABLE household_capabilities ENABLE ROW LEVEL SECURITY;
ALTER TABLE household_consent_defaults ENABLE ROW LEVEL SECURITY;
ALTER TABLE action_playbooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE action_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE incident_packets ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS household_capabilities_all ON household_capabilities;
CREATE POLICY household_capabilities_all ON household_capabilities
  FOR ALL USING (household_id = public.user_household_id());

DROP POLICY IF EXISTS household_consent_defaults_all ON household_consent_defaults;
CREATE POLICY household_consent_defaults_all ON household_consent_defaults
  FOR ALL USING (household_id = public.user_household_id());

DROP POLICY IF EXISTS action_playbooks_all ON action_playbooks;
CREATE POLICY action_playbooks_all ON action_playbooks
  FOR ALL USING (household_id = public.user_household_id());

DROP POLICY IF EXISTS action_tasks_all ON action_tasks;
CREATE POLICY action_tasks_all ON action_tasks
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM action_playbooks ap
      WHERE ap.id = action_tasks.playbook_id AND ap.household_id = public.user_household_id()
    )
  );

DROP POLICY IF EXISTS incident_packets_all ON incident_packets;
CREATE POLICY incident_packets_all ON incident_packets
  FOR ALL USING (household_id = public.user_household_id());
