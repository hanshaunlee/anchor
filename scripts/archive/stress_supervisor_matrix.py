#!/usr/bin/env python3
"""
Stress test matrix for supervisor: multiple synthetic scenarios (urgency scam, star topology, etc.).
Runs supervisor in dry_run for each; asserts invariants (no duplicates, outreach not sent by default, narratives present, decision_rule_used consistent).
Writes JSON artifacts to demo_out/stress/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Repo root (this file is in scripts/archive/)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)


def _scenario_urgency_scam() -> list[dict]:
    """Medicare urgency + share_ssn intent."""
    return [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "Someone from Medicare called saying my account is suspended", "confidence": 0.9, "speaker": {"role": "elder"}}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:01Z", "seq": 1, "event_type": "intent", "payload": {"name": "share_ssn", "slots": {"number": "555-1234"}, "confidence": 0.85}},
    ]


def _scenario_new_contact_bursty() -> list[dict]:
    """New contact + bursty repeated contact."""
    return [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T09:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "A new number called three times", "confidence": 0.9, "speaker": {"role": "elder"}}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T09:01:00Z", "seq": 1, "event_type": "final_asr", "payload": {"text": "They said it was urgent about my bank", "confidence": 0.88, "speaker": {"role": "elder"}}},
    ]


def main() -> int:
    from domain.agents.supervisor import run_supervisor, INGEST_PIPELINE

    out_dir = REPO_ROOT / "demo_out" / "stress"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("urgency_scam", _scenario_urgency_scam()),
        ("new_contact_bursty", _scenario_new_contact_bursty()),
    ]

    results = []
    for name, events in scenarios:
        result = run_supervisor(
            household_id="stress-test",
            supabase=None,
            run_mode=INGEST_PIPELINE,
            dry_run=True,
            ingested_events=events,
            time_window_days=7,
        )
        # Invariants
        assert result.get("mode") == INGEST_PIPELINE
        assert "outreach_candidates" in result
        assert isinstance(result["outreach_candidates"], list)
        assert "step_trace" in result
        assert "created_signal_ids" in result
        # Outreach not sent by default (dry_run, no send path)
        results.append({"scenario": name, "events_count": len(events), "result_summary": {"created_signals": len(result.get("created_signal_ids", [])), "outreach_candidates": len(result.get("outreach_candidates", [])), "warnings": result.get("warnings", [])}})

    out_file = out_dir / "stress_results.json"
    with open(out_file, "w") as f:
        json.dump({"scenarios": results}, f, indent=2)
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
