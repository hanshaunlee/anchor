#!/usr/bin/env python3
"""
One-time upload: load the 35k elderly conversation data into Supabase for the
user/household in config/demo_placeholder.json.

1. Put your Supabase user_id and household_id in config/demo_placeholder.json.
2. Generate data (if not already present): python3 scripts/generate_elderly_conversations.py
3. Run once (use the project venv so supabase is available):
   PYTHONPATH=apps/api:. .venv/bin/python3 scripts/load_elderly_conversations.py --demo

To load into a specific household without config: use --household-id <uuid>.

Requires: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY in .env or apps/api/.env.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _load_dotenv() -> None:
    for env_path in (_repo_root / "apps" / "api" / ".env", _repo_root / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if "#" in v and not v.startswith("'"):
                v = v.split("#")[0].strip()
            os.environ.setdefault(k.strip(), v)


def _get_demo_placeholder() -> dict[str, str] | None:
    from config.demo_placeholder import get_demo_placeholder
    return get_demo_placeholder()


def _ensure_devices(client, household_id: str, count: int = 3) -> list[str]:
    """Return at least `count` device IDs for the household; create if needed."""
    r = client.table("devices").select("id").eq("household_id", household_id).execute()
    existing = [d["id"] for d in (r.data or [])]
    if len(existing) >= count:
        return existing[:count]
    from uuid import uuid4
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    to_create = count - len(existing)
    new_ids = [str(uuid4()) for _ in range(to_create)]
    for did in new_ids:
        client.table("devices").insert({
            "id": did,
            "household_id": household_id,
            "device_type": "anchor_speaker",
            "firmware_version": "1.0.0",
            "last_seen_at": now,
        }).execute()
    return existing[:count] + new_ids


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Load elderly conversation data into Supabase (demo or given household).")
    parser.add_argument("--demo", action="store_true", help="Use demo household from config/demo_placeholder.json (or DEMO_HOUSEHOLD_ID)")
    parser.add_argument("--household-id", type=str, default=None, help="Target household UUID (use this or --demo)")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with sessions.json, event_packets.ndjson, summaries.json (default: data/elderly_conversations)")
    parser.add_argument("--dry-run", action="store_true", help="Print counts and mapping only; no DB writes")
    args = parser.parse_args()

    _load_dotenv()
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (e.g. in .env or apps/api/.env)", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data_dir) if args.data_dir else _repo_root / "data" / "elderly_conversations"
    if not data_dir.exists():
        print("Data directory not found:", data_dir, file=sys.stderr)
        print("Run first: python3 scripts/generate_elderly_conversations.py", file=sys.stderr)
        sys.exit(1)

    sessions_path = data_dir / "sessions.json"
    events_path = data_dir / "event_packets.ndjson"
    summaries_path = data_dir / "summaries.json"
    for p in (sessions_path, events_path, summaries_path):
        if not p.exists():
            print("Missing file:", p, file=sys.stderr)
            sys.exit(1)

    from supabase import create_client
    client = create_client(url, key)

    if args.household_id:
        household_id = args.household_id
        print("Using household:", household_id)
    elif args.demo:
        placeholder = _get_demo_placeholder()
        if not placeholder:
            print("Demo placeholder not set. Edit config/demo_placeholder.json with your user_id and household_id,", file=sys.stderr)
            print("or set env DEMO_USER_ID and DEMO_HOUSEHOLD_ID.", file=sys.stderr)
            sys.exit(1)
        household_id = placeholder["household_id"]
        print("Using demo household from config:", household_id)
    else:
        print("Provide --household-id <uuid> or --demo (requires config/demo_placeholder.json)", file=sys.stderr)
        sys.exit(1)

    with open(sessions_path) as f:
        blob = json.load(f)
    file_household_id = blob.get("household_id")
    file_device_ids = list(blob.get("device_ids", []))
    sessions = blob.get("sessions", [])

    with open(summaries_path) as f:
        sum_blob = json.load(f)
    summaries = sum_blob.get("summaries", [])

    # Map file device_id -> target device_id (by index)
    if args.dry_run:
        target_device_ids = file_device_ids  # identity map; no DB
    else:
        target_device_ids = _ensure_devices(client, household_id, max(3, len(file_device_ids)))
        if len(file_device_ids) > len(target_device_ids):
            while len(target_device_ids) < len(file_device_ids):
                target_device_ids.append(target_device_ids[len(target_device_ids) % max(1, len(target_device_ids))])
    device_map = {}
    for i, fid in enumerate(file_device_ids):
        device_map[fid] = target_device_ids[i % len(target_device_ids)]

    if args.dry_run:
        print("[dry-run] Sessions:", len(sessions))
        print("[dry-run] Summaries:", len(summaries))
        print("[dry-run] Device mapping:", list(device_map.items())[:6], "...")
        with open(events_path) as ef:
            n_events = sum(1 for _ in ef)
        print("[dry-run] Events:", n_events)
        return

    # Insert sessions (keep id from file so events match)
    session_rows = []
    for s in sessions:
        session_rows.append({
            "id": s["id"],
            "household_id": household_id,
            "device_id": device_map.get(s["device_id"], target_device_ids[0]),
            "started_at": s["started_at"],
            "ended_at": s["ended_at"],
            "mode": s.get("mode", "offline"),
            "consent_state": s.get("consent_state", {}),
        })
    print("Upserting", len(session_rows), "sessions (idempotent: re-run skips existing)...")
    BATCH = 100
    for i in range(0, len(session_rows), BATCH):
        chunk = session_rows[i : i + BATCH]
        client.table("sessions").upsert(chunk, on_conflict="id").execute()
        if (i + BATCH) % 5000 == 0 or i + BATCH >= len(session_rows):
            print("  ", min(i + BATCH, len(session_rows)), "/", len(session_rows))

    # Upsert events (idempotent: session_id+seq unique)
    print("Upserting events...")
    event_batch = []
    inserted = 0
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            e["device_id"] = device_map.get(e["device_id"], target_device_ids[0])
            event_batch.append({
                "session_id": e["session_id"],
                "device_id": e["device_id"],
                "ts": e["ts"],
                "seq": e["seq"],
                "event_type": e["event_type"],
                "payload": e.get("payload", {}),
                "payload_version": e.get("payload_version", 1),
            })
            if len(event_batch) >= 500:
                client.table("events").upsert(event_batch, on_conflict="session_id,seq").execute()
                inserted += len(event_batch)
                if inserted % 10000 == 0:
                    print("  ", inserted, "events")
                event_batch = []
        if event_batch:
            client.table("events").upsert(event_batch, on_conflict="session_id,seq").execute()
            inserted += len(event_batch)
    print("  Total events:", inserted)

    # Insert summaries (session-scoped; household_id = target)
    summary_rows = []
    for s in summaries:
        summary_rows.append({
            "household_id": household_id,
            "session_id": s.get("session_id"),
            "period_start": None,
            "period_end": None,
            "summary_text": s.get("summary_text", ""),
            "summary_json": s.get("summary_json", {}),
        })
    print("Inserting", len(summary_rows), "summaries...")
    for i in range(0, len(summary_rows), 500):
        client.table("summaries").insert(summary_rows[i : i + 500]).execute()
    print("Done. Household:", household_id)
    print("Run pipeline: python -m worker.main --household-id", household_id, "--once")

if __name__ == "__main__":
    main()
