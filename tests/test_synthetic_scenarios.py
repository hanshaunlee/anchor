"""Tests for scripts.synthetic_scenarios: make_ts, baseline_normal_events, scam_scenario_events."""
from datetime import datetime, timezone

import pytest

from scripts.synthetic_scenarios import make_ts, baseline_normal_events, scam_scenario_events


def test_make_ts_format() -> None:
    s = make_ts(day_offset=0, hour=10, minute=30)
    assert "T" in s or " " in s
    parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
    assert parsed.hour == 10
    assert parsed.minute == 30


def test_baseline_normal_events_structure() -> None:
    events = baseline_normal_events("s1", "d1", day_offset=1)
    assert len(events) >= 2
    for ev in events:
        assert ev["session_id"] == "s1"
        assert ev["device_id"] == "d1"
        assert "ts" in ev
        assert "seq" in ev
        assert "event_type" in ev
        assert "payload" in ev
    types = [e["event_type"] for e in events]
    assert "wake" in types
    assert "final_asr" in types or "intent" in types


def test_baseline_normal_events_reminder_content() -> None:
    events = baseline_normal_events("s1", "d1")
    asr = next((e for e in events if e.get("event_type") == "final_asr"), None)
    assert asr is not None
    assert "Remind" in (asr.get("payload") or {}).get("text", "")


def test_scam_scenario_events_structure() -> None:
    events = scam_scenario_events("s1", "d1", day_offset=0)
    assert len(events) >= 3
    for ev in events:
        assert ev["session_id"] == "s1"
        assert ev["device_id"] == "d1"
    types = [e["event_type"] for e in events]
    assert "final_asr" in types
    assert "intent" in types


def test_scam_scenario_events_urgency_content() -> None:
    events = scam_scenario_events("s1", "d1")
    texts = " ".join(
        (e.get("payload") or {}).get("text", "")
        for e in events
        if e.get("event_type") == "final_asr"
    )
    assert "Medicare" in texts or "Social Security" in texts or "ssn" in texts.lower()
