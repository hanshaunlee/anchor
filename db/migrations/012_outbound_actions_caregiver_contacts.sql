-- Outbound actions (caregiver notify/call/email) and caregiver_contacts.
-- Enums for action/channel/status/provider.

CREATE TYPE outbound_action_type AS ENUM ('caregiver_notify', 'caregiver_call', 'caregiver_email');
CREATE TYPE outbound_channel AS ENUM ('sms', 'email', 'voice_call');
CREATE TYPE outbound_action_status AS ENUM ('queued', 'sent', 'delivered', 'failed', 'suppressed');
CREATE TYPE outbound_provider AS ENUM ('twilio', 'sendgrid', 'smtp', 'mock');

CREATE TABLE outbound_actions (
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

CREATE INDEX idx_outbound_actions_household ON outbound_actions(household_id);
CREATE INDEX idx_outbound_actions_risk_signal ON outbound_actions(triggered_by_risk_signal_id);
CREATE INDEX idx_outbound_actions_created ON outbound_actions(created_at DESC);

CREATE TABLE caregiver_contacts (
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

CREATE INDEX idx_caregiver_contacts_household ON caregiver_contacts(household_id);

ALTER TABLE outbound_actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE caregiver_contacts ENABLE ROW LEVEL SECURITY;

-- Outbound actions: any household member can SELECT (API redacts by role: elder sees elder_safe_message only).
CREATE POLICY outbound_actions_select_household ON outbound_actions
  FOR SELECT USING (household_id = public.user_household_id());
CREATE POLICY outbound_actions_insert_caregiver_admin ON outbound_actions
  FOR INSERT WITH CHECK (
    household_id = public.user_household_id() AND public.user_can_contact()
  );
CREATE POLICY outbound_actions_update_caregiver_admin ON outbound_actions
  FOR UPDATE USING (
    household_id = public.user_household_id() AND public.user_can_contact()
  );

-- Caregiver contacts: only caregivers/admin read/write; elder could be given "names only" via a view later.
CREATE POLICY caregiver_contacts_all_caregiver_admin ON caregiver_contacts
  FOR ALL USING (
    household_id = public.user_household_id() AND public.user_can_contact()
  );
