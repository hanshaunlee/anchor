-- RPCs for compound reads: reduce round trips for alert detail and investigation window.
-- Call from API: supabase.rpc('rpc_get_alert_page_context', { p_signal_id, p_household_id })
--               supabase.rpc('rpc_get_investigation_window_context', { p_household_id, p_window_days })

-- Alert page context: consent, capabilities, last outbound for signal, caregiver contacts exist.
-- One call instead of 4–5 (sessions consent, household_capabilities, outbound_actions, caregiver_contacts).
CREATE OR REPLACE FUNCTION public.rpc_get_alert_page_context(
  p_signal_id UUID,
  p_household_id UUID
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_consent_state JSONB := '{}';
  v_capabilities JSONB := '{}';
  v_last_outbound JSONB := 'null';
  v_caregiver_contacts_exist BOOLEAN := false;
BEGIN
  -- Latest session consent for household
  SELECT COALESCE(s.consent_state, '{}')
  INTO v_consent_state
  FROM sessions s
  WHERE s.household_id = p_household_id
  ORDER BY s.started_at DESC
  LIMIT 1;

  -- Household capabilities (one row)
  SELECT to_jsonb(hc.*)
  INTO v_capabilities
  FROM household_capabilities hc
  WHERE hc.household_id = p_household_id
  LIMIT 1;
  IF v_capabilities IS NULL THEN
    v_capabilities := '{}';
  END IF;

  -- Last outbound action for this signal
  SELECT jsonb_build_object(
    'status', oa.status,
    'created_at', oa.created_at,
    'sent_at', oa.sent_at
  )
  INTO v_last_outbound
  FROM outbound_actions oa
  WHERE oa.household_id = p_household_id
    AND oa.triggered_by_risk_signal_id = p_signal_id
  ORDER BY oa.created_at DESC
  LIMIT 1;
  IF v_last_outbound IS NULL THEN
    v_last_outbound := 'null';
  END IF;

  -- Caregiver contacts exist for household
  SELECT EXISTS (
    SELECT 1 FROM caregiver_contacts cc
    WHERE cc.household_id = p_household_id
    LIMIT 1
  ) INTO v_caregiver_contacts_exist;

  RETURN jsonb_build_object(
    'consent_state', v_consent_state,
    'capabilities', v_capabilities,
    'last_outbound_for_signal', v_last_outbound,
    'caregiver_contacts_exist', v_caregiver_contacts_exist
  );
END;
$$;

COMMENT ON FUNCTION public.rpc_get_alert_page_context(UUID, UUID) IS
  'Compound read for alert detail page: consent, capabilities, last outbound action for signal, caregiver contacts exist. Reduces 4–8 round trips to one.';

-- Investigation window context: last run metadata + events signature for household in window.
CREATE OR REPLACE FUNCTION public.rpc_get_investigation_window_context(
  p_household_id UUID,
  p_window_days INT DEFAULT 7
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_since TIMESTAMPTZ;
  v_last_run_at TIMESTAMPTZ;
  v_last_run_id UUID;
  v_events_count BIGINT := 0;
  v_max_event_ts TIMESTAMPTZ;
BEGIN
  v_since := now() - (p_window_days || ' days')::interval;

  -- Last agent run for this household (e.g. financial or supervisor-style; we use agent_name like 'financial_security')
  SELECT ar.started_at, ar.id
  INTO v_last_run_at, v_last_run_id
  FROM agent_runs ar
  WHERE ar.household_id = p_household_id
  ORDER BY ar.started_at DESC
  LIMIT 1;

  -- Events in window (sessions belong to household via device)
  SELECT COUNT(e.id), MAX(e.ts)
  INTO v_events_count, v_max_event_ts
  FROM events e
  JOIN sessions s ON s.id = e.session_id
  WHERE s.household_id = p_household_id
    AND e.ts >= v_since;

  RETURN jsonb_build_object(
    'last_run_at', v_last_run_at,
    'last_run_id', v_last_run_id,
    'events_count_in_window', COALESCE(v_events_count, 0),
    'max_event_ts', v_max_event_ts,
    'window_days', p_window_days
  );
END;
$$;

COMMENT ON FUNCTION public.rpc_get_investigation_window_context(UUID, INT) IS
  'Compound read for investigation window: last run metadata + events count/max_ts in window. Used by API to decide run_ingest_pipeline.';
