-- Action playbooks, action_tasks, incident_packets, household_capabilities.
-- Consent defaults and escalation at household level; sessions.consent_state remains per-session override.

-- Enums for playbooks and tasks
CREATE TYPE playbook_type AS ENUM ('bank_contact', 'account_lockdown', 'device_high_risk_mode', 'incident_response');
CREATE TYPE playbook_status AS ENUM ('active', 'completed', 'canceled');
CREATE TYPE action_task_type AS ENUM (
  'call_bank', 'email_bank', 'enable_alerts', 'freeze_card_instruction',
  'change_password_instruction', 'device_high_risk_mode_push',
  'verify_with_elder', 'notify_caregiver', 'file_report', 'lock_card'
);
CREATE TYPE action_task_status AS ENUM ('ready', 'blocked', 'done', 'skipped');
CREATE TYPE bank_data_connector_enum AS ENUM ('none', 'plaid', 'open_banking', 'custom');

-- Household capabilities (explicit registry: what this household can do)
CREATE TABLE household_capabilities (
  household_id UUID PRIMARY KEY REFERENCES households(id) ON DELETE CASCADE,
  notify_sms_enabled BOOLEAN NOT NULL DEFAULT false,
  notify_email_enabled BOOLEAN NOT NULL DEFAULT false,
  device_policy_push_enabled BOOLEAN NOT NULL DEFAULT true,
  bank_data_connector bank_data_connector_enum NOT NULL DEFAULT 'none',
  bank_control_capabilities JSONB NOT NULL DEFAULT '{"lock_card": false, "disable_transfers": false, "enable_alerts": true, "open_dispute": false}',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_household_capabilities_household ON household_capabilities(household_id);

-- Household consent defaults (elder/caregiver set during onboarding; sessions.consent_state overrides per session)
CREATE TABLE household_consent_defaults (
  household_id UUID PRIMARY KEY REFERENCES households(id) ON DELETE CASCADE,
  share_with_caregiver BOOLEAN NOT NULL DEFAULT true,
  share_text BOOLEAN NOT NULL DEFAULT true,
  allow_outbound_contact BOOLEAN NOT NULL DEFAULT false,
  escalation_threshold INT NOT NULL DEFAULT 3,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT escalation_threshold_range CHECK (escalation_threshold >= 1 AND escalation_threshold <= 5)
);

CREATE INDEX idx_household_consent_defaults_household ON household_consent_defaults(household_id);

-- Action playbooks (DAG of steps for an incident)
CREATE TABLE action_playbooks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  risk_signal_id UUID NOT NULL REFERENCES risk_signals(id) ON DELETE CASCADE,
  playbook_type playbook_type NOT NULL,
  graph JSONB NOT NULL DEFAULT '{"nodes": [], "edges": []}',
  status playbook_status NOT NULL DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_action_playbooks_household ON action_playbooks(household_id);
CREATE INDEX idx_action_playbooks_risk_signal ON action_playbooks(risk_signal_id);
CREATE INDEX idx_action_playbooks_status ON action_playbooks(status);

-- Action tasks (one per playbook step; details may contain phone/email — RLS redacts for elder)
CREATE TABLE action_tasks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  playbook_id UUID NOT NULL REFERENCES action_playbooks(id) ON DELETE CASCADE,
  task_type action_task_type NOT NULL,
  status action_task_status NOT NULL DEFAULT 'ready',
  details JSONB NOT NULL DEFAULT '{}',
  completed_by_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_action_tasks_playbook ON action_tasks(playbook_id);
CREATE INDEX idx_action_tasks_status ON action_tasks(status);

-- Incident packets (bank-ready, evidence-cited case file)
CREATE TABLE incident_packets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  household_id UUID NOT NULL REFERENCES households(id) ON DELETE CASCADE,
  risk_signal_id UUID NOT NULL REFERENCES risk_signals(id) ON DELETE CASCADE,
  packet_json JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_incident_packets_household ON incident_packets(household_id);
CREATE INDEX idx_incident_packets_risk_signal ON incident_packets(risk_signal_id);

-- RLS
ALTER TABLE household_capabilities ENABLE ROW LEVEL SECURITY;
ALTER TABLE household_consent_defaults ENABLE ROW LEVEL SECURITY;
ALTER TABLE action_playbooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE action_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE incident_packets ENABLE ROW LEVEL SECURITY;

CREATE POLICY household_capabilities_all ON household_capabilities
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY household_consent_defaults_all ON household_consent_defaults
  FOR ALL USING (household_id = public.user_household_id());

CREATE POLICY action_playbooks_all ON action_playbooks
  FOR ALL USING (household_id = public.user_household_id());

-- action_tasks: elder can SELECT but API redacts details.phone/email for elder (see service layer).
CREATE POLICY action_tasks_all ON action_tasks
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM action_playbooks ap
      WHERE ap.id = action_tasks.playbook_id AND ap.household_id = public.user_household_id()
    )
  );

CREATE POLICY incident_packets_all ON incident_packets
  FOR ALL USING (household_id = public.user_household_id());

COMMENT ON TABLE household_capabilities IS 'Explicit capability registry: notify, device push, bank connector and control flags. Never overpromise.';
COMMENT ON TABLE action_playbooks IS 'Incident response playbook DAG; status active|completed|canceled.';
COMMENT ON TABLE action_tasks IS 'Per-step tasks; details may contain contact info — redact for elder role.';
COMMENT ON TABLE incident_packets IS 'Bank-ready case file (evidence_refs only); export for caregiver.';
