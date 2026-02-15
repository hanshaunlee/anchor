#!/usr/bin/env python3
"""
Caregiver Outreach Agent demo: generate one risk signal (or use existing), then run outreach.
Uses MockProvider (logs + optional DB). For real DB set SUPABASE_URL and run migrations 011/012.
"""
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "apps" / "api"))


def main() -> None:
    from unittest.mock import MagicMock

    from config.demo_placeholder import get_demo_placeholder
    from domain.agents.caregiver_outreach_agent import run_caregiver_outreach_agent
    from domain.consent import normalize_consent_state

    ph = get_demo_placeholder()
    household_id = (ph.get("household_id") or "demo-household") if ph else "demo-household"
    risk_signal_id = str(uuid4())
    session_id = str(uuid4())

    signal_row = {
        "id": risk_signal_id,
        "household_id": household_id,
        "explanation": {
            "summary": "Possible urgency + request to share SSN. Medicare-themed phrasing.",
            "session_ids": [session_id],
            "motif_tags": ["new_contact", "urgency"],
            "timeline_snippet": [{"ts": "2025-01-15T10:00:00Z", "event_type": "final_asr", "text_preview": "(redacted)"}],
            "model_available": False,
        },
        "recommended_action": {"checklist": ["Review in dashboard", "Consider calling to check in"]},
        "severity": 4,
    }

    consent = normalize_consent_state({"consent_allow_outbound_contact": True})
    contacts = [
        {
            "id": str(uuid4()),
            "household_id": household_id,
            "name": "Demo Caregiver",
            "relationship": "family",
            "channels": {"sms": {"number": "+1555****", "last4": "0000"}, "email": {"email": "caregiver@example.com"}},
            "priority": 1,
            "quiet_hours": {"start": "22:00", "end": "08:00"},
        }
    ]

    mock_sb = MagicMock()
    risk_chain = MagicMock()
    risk_chain.select.return_value = risk_chain
    risk_chain.eq.return_value = risk_chain
    risk_chain.single.return_value = risk_chain
    risk_chain.execute.return_value.data = signal_row

    def table_fix(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.limit.return_value = t
        t.single.return_value = t
        t.in_.return_value = t
        t.insert.return_value.execute.return_value.data = [{"id": str(uuid4())}]
        t.update.return_value.eq.return_value.execute.return_value = None
        if name == "risk_signals":
            return risk_chain
        if name == "sessions":
            t.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
                {"consent_state": {"consent_allow_outbound_contact": True}}
            ]
        if name == "caregiver_contacts":
            t.select.return_value.eq.return_value.order.return_value.execute.return_value.data = contacts
        t.execute.return_value.data = []
        return t
    mock_sb.table.side_effect = table_fix

    print("Caregiver Outreach Agent — demo (MockProvider)")
    print("Running playbook for one risk signal…")
    result = run_caregiver_outreach_agent(
        household_id,
        mock_sb,
        risk_signal_id=risk_signal_id,
        dry_run=False,
        consent_state=consent,
        user_role="caregiver",
    )
    print("Step trace:")
    for s in result.get("step_trace") or []:
        print(f"  {s.get('step')}: {s.get('status')} — {s.get('notes', '')}")
    print("Summary:", result.get("summary_json"))
    print("Outbound action ID:", result.get("outbound_action_id"))
    print("Done. In production, use POST /actions/outreach or POST /agents/outreach/run (caregiver/admin).")


if __name__ == "__main__":
    main()
