"""
Canonical demo user and household for this project.
Used by: seed_supabase_data.py, load_elderly_conversations.py, run_outreach_demo.py,
run_financial_agent_demo.py, demo_replay.py, and docs.

Resolution order: env DEMO_USER_ID / DEMO_HOUSEHOLD_ID, then config/demo_placeholder.json.
Replace the placeholder UUIDs in demo_placeholder.json with your real Supabase user id
and household id to load data and run demos against that account.
"""
from __future__ import annotations

import json
from pathlib import Path

_config_dir = Path(__file__).resolve().parent
_DEFAULT_PATH = _config_dir / "demo_placeholder.json"


def get_demo_placeholder() -> dict[str, str] | None:
    """Return {"user_id": "<uuid>", "household_id": "<uuid>"} or None if not configured."""
    import os
    uid = os.environ.get("DEMO_USER_ID", "").strip()
    hid = os.environ.get("DEMO_HOUSEHOLD_ID", "").strip()
    if uid and hid:
        return {"user_id": uid, "household_id": hid}
    path = os.environ.get("DEMO_PLACEHOLDER_PATH", str(_DEFAULT_PATH))
    try:
        with open(path) as f:
            data = json.load(f)
        uid = (data.get("user_id") or "").strip()
        hid = (data.get("household_id") or "").strip()
        if uid and hid and uid != "00000000-0000-0000-0000-000000000000" and hid != "00000000-0000-0000-0000-000000000000":
            return {"user_id": uid, "household_id": hid}
    except Exception:
        pass
    return None
