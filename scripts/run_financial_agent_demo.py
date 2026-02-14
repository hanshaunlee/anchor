#!/usr/bin/env python3
"""
Financial Security Agent demo script (no API, no DB).
Runs the playbook on built-in demo events and prints summary.
For full demo harness (risk chart, explanation subgraph, agent trace JSON, optional UI),
use: python scripts/demo_replay.py [--ui] [--launch-ui]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "apps" / "api"))

from api.agents.financial_agent import get_demo_events, run_financial_security_playbook


def main() -> None:
    print("Financial Security Agent — demo (dry run)")
    print("Running playbook on built-in demo events…")
    events = get_demo_events()
    result = run_financial_security_playbook(
        household_id="demo",
        time_window_days=7,
        consent_state={"share_with_caregiver": True, "watchlist_ok": True},
        ingested_events=events,
        supabase=None,
        dry_run=True,
    )
    print(f"Events: {len(events)}, Risk signals: {len(result.get('risk_signals', []))}, Watchlists: {len(result.get('watchlists', []))}")
    for log in result.get("logs", []):
        print(f"  {log}")
    print("Done. For one-command demo with artifacts and UI: python scripts/demo_replay.py --ui --launch-ui")


if __name__ == "__main__":
    main()
