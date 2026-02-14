#!/usr/bin/env python3
"""
Run agents on a schedule (drift weekly, calibration weekly, ring daily, narratives hourly, redteam daily).
Use from cron or worker; dry_run in dev to avoid writes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "api"))

DRY_RUN = os.environ.get("ANCHOR_AGENT_CRON_DRY_RUN", "1").lower() in ("1", "true", "yes")


def get_supabase_client():
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY", "")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None


def run_drift(household_id: str, supabase, dry_run: bool) -> dict:
    from domain.agents.graph_drift_agent import run_graph_drift_agent
    return run_graph_drift_agent(household_id, supabase=supabase, dry_run=dry_run)


def run_narrative(household_id: str, supabase, dry_run: bool) -> dict:
    from domain.agents.evidence_narrative_agent import run_evidence_narrative_agent
    return run_evidence_narrative_agent(household_id, supabase=supabase, dry_run=dry_run)


def run_ring(household_id: str, supabase, dry_run: bool) -> dict:
    from domain.agents.ring_discovery_agent import run_ring_discovery_agent
    return run_ring_discovery_agent(household_id, supabase=supabase, neo4j_available=False, dry_run=dry_run)


def run_calibration(household_id: str, supabase, dry_run: bool) -> dict:
    from domain.agents.continual_calibration_agent import run_continual_calibration_agent
    return run_continual_calibration_agent(household_id, supabase=supabase, dry_run=dry_run)


def run_redteam(household_id: str, supabase, dry_run: bool) -> dict:
    from domain.agents.synthetic_redteam_agent import run_synthetic_redteam_agent
    return run_synthetic_redteam_agent(household_id, supabase=supabase, dry_run=dry_run)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run scheduled agents")
    p.add_argument("--household-id", required=True, help="Household UUID")
    p.add_argument("--agent", choices=["drift", "narrative", "ring", "calibration", "redteam"], required=True)
    p.add_argument("--no-dry-run", action="store_true", help="Persist to DB (default: dry run)")
    args = p.parse_args()
    dry_run = not args.no_dry_run or DRY_RUN
    supabase = get_supabase_client()
    if not supabase:
        print("No Supabase client; skipping persist.", file=sys.stderr)
        dry_run = True
    runners = {
        "drift": run_drift,
        "narrative": run_narrative,
        "ring": run_ring,
        "calibration": run_calibration,
        "redteam": run_redteam,
    }
    result = runners[args.agent](args.household_id, supabase, dry_run)
    print("status:", result.get("status"))
    print("run_id:", result.get("run_id"))
    if result.get("summary_json"):
        print("summary:", result["summary_json"])


if __name__ == "__main__":
    main()
