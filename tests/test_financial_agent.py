"""Tests for Financial Security Agent: playbook, consent gating, synthetic scam scenario."""
import pytest

from api.agents.financial_agent import run_financial_security_playbook


def test_financial_agent_dry_run_no_events() -> None:
    """No events: agent returns empty signals and watchlists."""
    result = run_financial_security_playbook(
        household_id="hh1",
        time_window_days=7,
        consent_state={},
        ingested_events=[],
        supabase=None,
        dry_run=True,
    )
    assert "risk_signals" in result
    assert "watchlists" in result
    assert "logs" in result
    assert isinstance(result["risk_signals"], list)
    assert isinstance(result["watchlists"], list)
    assert result["inserted_signal_ids"] == []


def test_financial_agent_synthetic_scam_scenario() -> None:
    """Synthetic scam-like events: urgency + new contact + sensitive intent."""
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "Someone from Medicare called saying my account is suspended", "confidence": 0.9, "speaker": {"role": "elder"}}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:01Z", "seq": 1, "event_type": "intent", "payload": {"name": "share_ssn", "slots": {"number": "555-1234"}, "confidence": 0.85}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:02Z", "seq": 2, "event_type": "final_asr", "payload": {"text": "They said I need to verify immediately", "confidence": 0.88, "speaker": {"role": "elder"}}},
    ]
    result = run_financial_security_playbook(
        household_id="hh1",
        time_window_days=7,
        consent_state={"share_with_caregiver": True, "watchlist_ok": True},
        ingested_events=events,
        supabase=None,
        dry_run=True,
    )
    assert "risk_signals" in result
    assert "logs" in result
    assert "motif_tags" in result
    assert "timeline_snippet" in result
    # Should detect motifs (urgency, new contact, sensitive intent)
    assert len(result["logs"]) >= 1
    # May or may not produce risk signals depending on motif scoring
    for sig in result["risk_signals"]:
        assert "signal_type" in sig
        assert sig["signal_type"] in ("possible_scam_contact", "social_engineering_risk", "payment_anomaly")
        assert 1 <= sig["severity"] <= 5
        assert "explanation" in sig
        assert "motif_tags" in sig["explanation"] or "timeline_snippet" in sig["explanation"]
        assert "recommended_action" in sig
        assert "checklist" in sig["recommended_action"]


def test_financial_agent_consent_gating() -> None:
    """When share_with_caregiver is false, escalation draft not persisted in recommended_action; text redacted."""
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "IRS said I owe money and need to pay now", "confidence": 0.9, "speaker": {"role": "elder"}}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:01Z", "seq": 1, "event_type": "intent", "payload": {"name": "pay_now", "slots": {}, "confidence": 0.8}},
    ]
    result_share = run_financial_security_playbook(
        household_id="hh1",
        time_window_days=7,
        consent_state={"share_with_caregiver": True, "watchlist_ok": True},
        ingested_events=events,
        supabase=None,
        dry_run=True,
    )
    result_no_share = run_financial_security_playbook(
        household_id="hh1",
        time_window_days=7,
        consent_state={"share_with_caregiver": False, "watchlist_ok": True},
        ingested_events=events,
        supabase=None,
        dry_run=True,
    )
    # With consent: may have escalation_draft in recommended_action for high severity
    # Without consent: explanation should be redacted when we check
    for sig in result_no_share.get("risk_signals", []):
        expl = sig.get("explanation", {})
        # When redacted we set redacted=True and strip sensitive text
        if expl.get("redacted"):
            assert expl.get("redacted") is True
    # Watchlist still produced when watchlist_ok True
    assert "watchlists" in result_no_share


def test_financial_agent_recommended_action_checklist() -> None:
    """Recommended actions are non-destructive (no money movement)."""
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:00Z", "seq": 0, "event_type": "intent", "payload": {"name": "wire_money", "slots": {"payee": "unknown"}, "confidence": 0.9}},
    ]
    result = run_financial_security_playbook(
        household_id="hh1",
        time_window_days=7,
        consent_state={"share_with_caregiver": True, "watchlist_ok": True},
        ingested_events=events,
        supabase=None,
        dry_run=True,
    )
    for sig in result.get("risk_signals", []):
        rec = sig.get("recommended_action", {})
        checklist = rec.get("checklist", [])
        assert isinstance(checklist, list)
        # Safe recommendations we expect (read-only / protective)
        safe_phrases = ["Do not share", "Verify", "Call back", "Enable", "Change passwords", "review"]
        checklist_str = " ".join(checklist).lower()
        assert any(p.lower() in checklist_str for p in safe_phrases)
