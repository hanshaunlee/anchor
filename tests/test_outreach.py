"""Tests: caregiver outreach agent and API — role enforcement, consent gating, quiet hours, mock provider, redaction."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("supabase")
from fastapi.testclient import TestClient

from api.main import app
from api.deps import get_supabase, require_user
from domain.agents.caregiver_outreach_agent import run_caregiver_outreach_agent, _in_quiet_hours, _generate_message_template
from domain.consent import normalize_consent_state
from domain.notify.providers import MockProvider, send_via_provider


# --- Consent normalization ---
def test_normalize_consent_state_defaults() -> None:
    out = normalize_consent_state({})
    assert "consent_allow_outbound_contact" in out
    assert "caregiver_contact_policy" in out
    assert "allowed_channels" in out.get("caregiver_contact_policy", {})


def test_normalize_consent_state_explicit_outbound() -> None:
    out = normalize_consent_state({"consent_allow_outbound_contact": True})
    assert out["consent_allow_outbound_contact"] is True


# --- Quiet hours ---
def test_in_quiet_hours_empty() -> None:
    assert _in_quiet_hours({}, datetime.now(timezone.utc)) is False


def test_in_quiet_hours_night() -> None:
    # 23:00 is between 22:00 and 08:00
    night = datetime(2025, 1, 15, 23, 0, 0, tzinfo=timezone.utc)
    assert _in_quiet_hours({"start": "22:00", "end": "08:00"}, night) is True


def test_in_quiet_hours_day() -> None:
    noon = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert _in_quiet_hours({"start": "22:00", "end": "08:00"}, noon) is False


# --- Mock provider ---
def test_mock_provider_send_sms_returns_message_id_and_status() -> None:
    p = MockProvider()
    r = p.send_sms("+15551234567", "Test message")
    assert r.success is True
    assert r.provider_message_id is not None
    assert r.provider_message_id.startswith("mock-")
    assert r.error is None
    assert r.status == "sent"


def test_mock_provider_send_email() -> None:
    r = send_via_provider("mock", "email", "caregiver@example.com", subject="Test", body="Body")
    assert r.success is True
    assert r.provider_message_id is not None
    assert r.status == "sent"


# --- Redaction: consent_share_text=false -> payload no raw caregiver text except elder_safe ---
def test_generate_message_template_redacted() -> None:
    signal = {"severity": 4, "explanation": {"summary": "Sensitive summary"}, "recommended_action": {}}
    evidence = {"recommended_actions": [], "evidence_refs": []}
    out = _generate_message_template(signal, evidence, consent_share_text=False)
    assert "caregiver_message" in out
    assert "elder_safe_message" in out
    assert "Details withheld" in out["caregiver_message"] or "consent" in out["caregiver_message"].lower() or "privacy" in out["caregiver_message"].lower() or "limited by privacy" in out["caregiver_message"].lower()


# --- Agent: consent gating -> suppressed, no provider call ---
def test_outreach_agent_consent_disallow_suppressed() -> None:
    mock_sb = MagicMock()
    signal = {
        "id": str(uuid4()),
        "household_id": "hh1",
        "explanation": {"summary": "Test", "session_ids": []},
        "recommended_action": {},
        "severity": 4,
    }

    risk_chain = MagicMock()
    risk_chain.select.return_value = risk_chain
    risk_chain.eq.return_value = risk_chain
    risk_chain.single.return_value = risk_chain
    risk_chain.execute.return_value.data = signal
    sess_chain = MagicMock()
    sess_chain.select.return_value = sess_chain
    sess_chain.eq.return_value = sess_chain
    sess_chain.order.return_value = sess_chain
    sess_chain.limit.return_value = sess_chain
    sess_chain.execute.return_value.data = [{"consent_state": {"consent_allow_outbound_contact": False}}]
    contacts_chain = MagicMock()
    contacts_chain.select.return_value = contacts_chain
    contacts_chain.eq.return_value = contacts_chain
    contacts_chain.order.return_value = contacts_chain
    contacts_chain.execute.return_value.data = []
    insert_chain = MagicMock()
    insert_chain.execute.return_value.data = [{"id": str(uuid4())}]

    def table(name):
        if name == "risk_signals":
            return risk_chain
        if name == "sessions":
            return sess_chain
        if name == "caregiver_contacts":
            return contacts_chain
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.limit.return_value = t
        t.single.return_value = t
        t.insert.return_value = insert_chain
        t.update.return_value.eq.return_value.execute.return_value = None
        t.execute.return_value.data = []
        return t

    mock_sb.table.side_effect = table
    result = run_caregiver_outreach_agent(
        "hh1",
        mock_sb,
        risk_signal_id=signal["id"],
        dry_run=False,
        consent_state=normalize_consent_state({"consent_allow_outbound_contact": False}),
        user_role="caregiver",
    )
    assert result.get("summary_json", {}).get("suppressed") is True
    assert result.get("status") == "completed"


# --- API: elder cannot trigger outreach (403) ---
@pytest.fixture
def client_outreach():
    mock_sb = MagicMock()
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.limit.return_value = user_q
    user_q.execute.return_value.data = [{"household_id": "hh-1", "role": "elder"}]
    sess_q = MagicMock()
    sess_q.select.return_value = sess_q
    sess_q.eq.return_value = sess_q
    sess_q.order.return_value = sess_q
    sess_q.limit.return_value = sess_q
    sess_q.execute.return_value.data = [{"consent_state": {}}]
    risk_q = MagicMock()
    risk_q.select.return_value = risk_q
    risk_q.eq.return_value = risk_q
    risk_q.single.return_value.execute.return_value.data = {
        "id": str(uuid4()),
        "household_id": "hh-1",
        "explanation": {},
        "recommended_action": {},
        "severity": 4,
    }
    insert_out = MagicMock()
    insert_out.execute.return_value.data = [{"id": str(uuid4())}]

    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.limit.return_value = t
        t.single.return_value = t
        t.insert.return_value = insert_out
        t.update.return_value.eq.return_value.execute.return_value = None
        if name == "users":
            return user_q
        if name == "sessions":
            return sess_q
        if name == "risk_signals":
            return risk_q
        t.execute.return_value.data = []
        return t

    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-elder"
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_post_outreach_elder_returns_403(client_outreach: TestClient) -> None:
    """Elder must not be able to trigger outreach."""
    r = client_outreach.post(
        "/actions/outreach",
        json={"risk_signal_id": str(uuid4()), "dry_run": True},
    )
    assert r.status_code == 403
    assert "caregiver" in r.json().get("detail", "").lower() or "admin" in r.json().get("detail", "").lower()


def test_elder_cannot_get_outreach_summary(client_outreach: TestClient) -> None:
    """Elder must not access outreach summary (caregiver/admin only)."""
    r = client_outreach.get("/actions/outreach/summary")
    assert r.status_code == 403


@pytest.fixture
def client_outreach_caregiver():
    mock_sb = MagicMock()
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.limit.return_value = user_q
    user_q.execute.return_value.data = [{"household_id": "hh-1", "role": "caregiver"}]
    sess_q = MagicMock()
    sess_q.select.return_value = sess_q
    sess_q.eq.return_value = sess_q
    sess_q.order.return_value = sess_q
    sess_q.limit.return_value = sess_q
    sess_q.execute.return_value.data = [{"consent_state": {"consent_allow_outbound_contact": True}}]
    risk_q = MagicMock()
    risk_q.select.return_value = risk_q
    risk_q.eq.return_value = risk_q
    risk_q.single.return_value.execute.return_value.data = {
        "id": str(uuid4()),
        "household_id": "hh-1",
        "explanation": {"summary": "Test", "session_ids": []},
        "recommended_action": {},
        "severity": 4,
    }
    insert_out = MagicMock()
    insert_out.execute.return_value.data = [{"id": str(uuid4())}]

    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.limit.return_value = t
        t.single.return_value = t
        t.insert.return_value = insert_out
        t.update.return_value.eq.return_value.execute.return_value = None
        if name == "users":
            return user_q
        if name == "sessions":
            return sess_q
        if name == "risk_signals":
            return risk_q
        t.execute.return_value.data = []
        return t

    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-caregiver"
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_post_outreach_caregiver_dry_run(client_outreach_caregiver: TestClient) -> None:
    """Caregiver dry_run: no send; no outbound_actions row (preview only)."""
    risk_id = str(uuid4())
    r = client_outreach_caregiver.post(
        "/actions/outreach",
        json={"risk_signal_id": risk_id, "dry_run": True},
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert data.get("outbound_action") is None
    assert "preview" in data or "suppressed" in data or "sent" in data


def test_get_outreach_summary_caregiver(client_outreach_caregiver: TestClient) -> None:
    """Caregiver can GET /actions/outreach/summary."""
    r = client_outreach_caregiver.get("/actions/outreach/summary")
    assert r.status_code == 200
    data = r.json()
    assert "counts" in data
    assert "recent" in data
    assert data["counts"].get("sent") is not None
    assert data["counts"].get("failed") is not None


def test_outreach_candidates_blocks_without_consent() -> None:
    """GET /actions/outreach/candidates returns candidates with blocking_reasons when consent outbound_contact_ok is false."""
    from api.deps import get_supabase, require_user, require_caregiver_or_admin

    risk_id = str(uuid4())
    queued_row = {
        "id": str(uuid4()),
        "triggered_by_risk_signal_id": risk_id,
        "status": "queued",
        "channel": "sms",
        "created_at": "2024-01-15T10:00:00Z",
        "payload": {},
    }
    mock_sb = MagicMock()
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.limit.return_value = user_q
    user_q.execute.return_value.data = [{"household_id": "hh-1", "role": "caregiver"}]
    sess_q = MagicMock()
    sess_q.select.return_value = sess_q
    sess_q.eq.return_value = sess_q
    sess_q.order.return_value = sess_q
    sess_q.limit.return_value = sess_q
    sess_q.execute.return_value.data = [{"consent_state": {"outbound_contact_ok": False, "share_with_caregiver": True}}]
    contacts_q = MagicMock()
    contacts_q.select.return_value = contacts_q
    contacts_q.eq.return_value = contacts_q
    contacts_q.limit.return_value = contacts_q
    contacts_q.execute.return_value.data = []
    oa_execute_calls = [[queued_row], []]

    def oa_execute():
        out = MagicMock()
        out.data = oa_execute_calls.pop(0) if oa_execute_calls else []
        return out

    oa_q = MagicMock()
    oa_q.select.return_value = oa_q
    oa_q.eq.return_value = oa_q
    oa_q.order.return_value = oa_q
    oa_q.limit.return_value = oa_q
    oa_q.in_.return_value = oa_q
    oa_q.execute.side_effect = oa_execute
    risk_sig_q = MagicMock()
    risk_sig_q.select.return_value = risk_sig_q
    risk_sig_q.eq.return_value = risk_sig_q
    risk_sig_q.in_.return_value = risk_sig_q
    risk_sig_q.execute.return_value.data = [{"id": risk_id, "severity": 4, "signal_type": "possible_scam_contact", "created_at": "2024-01-15T09:00:00Z", "status": "open"}]

    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.limit.return_value = t
        t.in_.return_value = t
        t.execute.return_value.data = []
        if name == "users":
            return user_q
        if name == "sessions":
            return sess_q
        if name == "caregiver_contacts":
            return contacts_q
        if name == "outbound_actions":
            return oa_q
        if name == "risk_signals":
            return risk_sig_q
        return t

    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-caregiver"
    app.dependency_overrides[require_caregiver_or_admin] = lambda: "user-caregiver"
    try:
        with TestClient(app) as client:
            r = client.get("/actions/outreach/candidates")
        assert r.status_code == 200
        data = r.json()
        assert "candidates" in data
        assert len(data["candidates"]) >= 1
        c = data["candidates"][0]
        assert c["risk_signal_id"] == risk_id
        assert c["consent_ok"] is False
        assert "consent_outbound" in c["blocking_reasons"]
        assert "outbound_contact_ok" in c.get("missing_consent_keys", [])
    finally:
        app.dependency_overrides.clear()
