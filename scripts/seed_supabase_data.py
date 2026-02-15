#!/usr/bin/env python3
"""
Seed Supabase with a few thousand data points for one user/household.
Anchor: elder/caregiver voice + financial security (scam detection, watchlists, risk signals).

Creates:
- 1 auth user (or uses SEED_USER_ID)
- 1 household, 1 user row, 2–3 devices
- ~150 sessions over ~180 days (varied timestamps)
- ~3000+ events (final_asr, intent, device_state, transaction_detected, payee_added, bank_alert_received)
- Utterances, entities, mentions, relationships, summaries
- Risk signals, watchlists, feedback, agent_runs, device_sync_state

Usage (from repo root):
  # Option A: Create new auth user + household + all data (may fail if auth settings restrict)
  PYTHONPATH=apps/api:. python scripts/seed_supabase_data.py

  # Option B: Use existing user (create in Supabase Dashboard → Authentication → Add user)
  SEED_USER_ID=<auth-users-uuid> PYTHONPATH=apps/api:. python scripts/seed_supabase_data.py

  # Option C: Add data only to an existing household (user already onboarded via app)
  PYTHONPATH=apps/api:. python scripts/seed_supabase_data.py --household-id <uuid> --user-id <auth-users-uuid>

  PYTHONPATH=apps/api:. python scripts/seed_supabase_data.py --dry-run   # print counts, no DB

Requires: .env or apps/api/.env with SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

# Repo root and apps/api for imports
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(_repo_root / "apps" / "api"))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _find_seed_user_by_email(client) -> str | None:
    try:
        r = client.auth.admin.list_users()
        users = getattr(r, "users", None) if r else None
        if not users and isinstance(r, (list, tuple)):
            users = r
        for u in (users or []):
            if getattr(u, "email", None) == SEED_EMAIL:
                return str(getattr(u, "id", u))
    except Exception:
        pass
    return None


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


# ---------- Data generation (elder / financial security context) ----------

SEED_EMAIL = "seed@anchor.demo"
SEED_PASSWORD = "AnchorSeedDemo1!"

# Entity canonicals we'll reuse (match graph builder: person, phone, org, merchant, topic, account)
PERSONS = ["Mom", "Dad", "Sarah", "John", "IRS Agent", "Medicare Rep", "Tech Support", "Grandkid", "Neighbor", "Doctor Smith"]
PHONES = ["+15551234567", "+15559876543", "+18005551234", "+15551112222", "555-1234", "800-633-4227"]
ORGS = ["IRS", "Medicare", "Social Security", "Amazon", "Apple", "Bank of America", "Wells Fargo", "Tech Support Inc"]
MERCHANTS = ["Amazon", "Walmart", "CVS", "Target", "Best Buy", "Home Depot", "Unknown Merchant", "Gift Card Co"]
TOPICS = ["Medicare", "Social Security", "tax refund", "gift cards", "account suspended", "verify identity", "urgent", "prize"]
ACCOUNT_HASHES = [f"acc_{hashlib.sha256(str(i).encode()).hexdigest()[:16]}" for i in range(20)]

# Utterances: routine + scam-like (elder voice)
ROUTINE_TEXTS = [
    "Remind me to take medicine at 5",
    "What's the weather today?",
    "Call Sarah",
    "Play some music",
    "Set a reminder for my doctor appointment",
    "Add milk to the shopping list",
    "When is my next pill?",
]
SCAM_LIKE_TEXTS = [
    "Someone from Medicare called saying my account is suspended",
    "They said I need to give my Social Security number to verify",
    "A man said he's from the IRS and I owe money",
    "They wanted me to buy gift cards to fix the computer",
    "Someone called about my car warranty expiring",
    "They said I won a prize and need to pay a fee",
    "Tech support said my computer has a virus",
]
INTENTS = ["reminder", "call_log", "share_ssn", "sensitive_request", "transfer_money", "buy_gift_cards", "verify_identity", "call_back"]


def _ts(day_offset: int, hour: int, minute: int, second: int = 0) -> str:
    t = datetime.now(timezone.utc) - timedelta(days=day_offset)
    t = t.replace(hour=hour, minute=minute, second=second, microsecond=0)
    return t.isoformat()


def _hash_canonical(s: str) -> str:
    return hashlib.sha256(s.strip().lower().encode()).hexdigest()[:16]


def generate_sessions(
    household_id: str,
    device_ids: list[str],
    num_sessions: int = 150,
    days_back: int = 180,
) -> list[dict]:
    sessions = []
    for i in range(num_sessions):
        day_off = random.randint(0, days_back)
        h = random.randint(6, 21)
        m = random.randint(0, 59)
        started = _ts(day_off, h, m, 0)
        duration_min = random.randint(2, 45)
        end_dt = datetime.fromisoformat(started.replace("Z", "+00:00")) + timedelta(minutes=duration_min)
        ended = end_dt.isoformat()
        device_id = random.choice(device_ids)
        mode = random.choice(["offline", "online"])
        sessions.append({
            "household_id": household_id,
            "device_id": device_id,
            "started_at": started,
            "ended_at": ended,
            "mode": mode,
            "consent_state": {},
        })
    return sessions


def generate_events_for_session(
    session_id: str,
    device_id: str,
    started_at: str,
    ended_at: str,
    session_index: int,
) -> list[dict]:
    """Generate 5–25 events per session: wake, final_asr, intent, device_state, and some financial events."""
    events = []
    start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    span_sec = max(60, (end_dt - start_dt).total_seconds())
    num_events = random.randint(5, min(25, int(span_sec / 15)))
    # Mix: ~60% routine, ~40% scam-like for variety
    is_scam_session = (session_index % 5) == 2 or random.random() < 0.2
    texts = SCAM_LIKE_TEXTS if is_scam_session else ROUTINE_TEXTS
    seq = 0
    t = start_dt
    step = span_sec / max(1, num_events)
    for i in range(num_events):
        ts = t + timedelta(seconds=i * step)
        if ts > end_dt:
            break
        ts_str = ts.isoformat()
        # wake
        if seq == 0:
            events.append({
                "session_id": session_id,
                "device_id": device_id,
                "ts": ts_str,
                "seq": seq,
                "event_type": "wake",
                "payload": {},
                "payload_version": 1,
            })
            seq += 1
            continue
        # device_state occasionally
        if random.random() < 0.15:
            events.append({
                "session_id": session_id,
                "device_id": device_id,
                "ts": ts_str,
                "seq": seq,
                "event_type": "device_state",
                "payload": {"online": True, "battery": round(random.uniform(0.5, 1.0), 2)},
                "payload_version": 1,
            })
            seq += 1
            continue
        # final_asr + intent
        if random.random() < 0.6:
            text = random.choice(texts)
            speaker = random.choice(["elder", "agent", "unknown"])
            events.append({
                "session_id": session_id,
                "device_id": device_id,
                "ts": ts_str,
                "seq": seq,
                "event_type": "final_asr",
                "payload": {
                    "text": text,
                    "confidence": round(random.uniform(0.75, 0.98), 2),
                    "speaker": {"role": speaker},
                },
                "payload_version": 1,
            })
            seq += 1
            intent_name = random.choice(INTENTS)
            slots = {}
            if "phone" in intent_name or random.random() < 0.3:
                slots["phone_number"] = random.choice(PHONES)
            if "topic" in intent_name or random.random() < 0.3:
                slots["topic"] = random.choice(TOPICS)
            events.append({
                "session_id": session_id,
                "device_id": device_id,
                "ts": (ts + timedelta(seconds=2)).isoformat(),
                "seq": seq,
                "event_type": "intent",
                "payload": {"name": intent_name, "slots": slots, "confidence": round(random.uniform(0.7, 0.95), 2)},
                "payload_version": 1,
            })
            seq += 1
            continue
        # financial events
        if random.random() < 0.25:
            which = random.choice(["transaction_detected", "payee_added", "bank_alert_received"])
            if which == "transaction_detected":
                events.append({
                    "session_id": session_id,
                    "device_id": device_id,
                    "ts": ts_str,
                    "seq": seq,
                    "event_type": "transaction_detected",
                    "payload": {
                        "merchant": random.choice(MERCHANTS),
                        "account_id_hash": random.choice(ACCOUNT_HASHES),
                        "confidence": 0.85,
                    },
                    "payload_version": 1,
                })
            elif which == "payee_added":
                events.append({
                    "session_id": session_id,
                    "device_id": device_id,
                    "ts": ts_str,
                    "seq": seq,
                    "event_type": "payee_added",
                    "payload": {
                        "payee_name": random.choice(PERSONS + MERCHANTS),
                        "payee_type": random.choice(["person", "merchant"]),
                        "confidence": 0.8,
                    },
                    "payload_version": 1,
                })
            else:
                events.append({
                    "session_id": session_id,
                    "device_id": device_id,
                    "ts": ts_str,
                    "seq": seq,
                    "event_type": "bank_alert_received",
                    "payload": {"account_id_hash": random.choice(ACCOUNT_HASHES), "confidence": 0.9},
                    "payload_version": 1,
                })
            seq += 1
    return events


def generate_entities(household_id: str) -> list[dict]:
    """Predefined entities for mentions/relationships."""
    entities = []
    seen = set()
    for name in PERSONS:
        h = _hash_canonical(name)
        if (name, h) in seen:
            continue
        seen.add((name, h))
        entities.append({"household_id": household_id, "entity_type": "person", "canonical": name, "canonical_hash": h, "meta": {}})
    for p in PHONES[:10]:
        h = _hash_canonical(p)
        if (p, h) in seen:
            continue
        seen.add((p, h))
        entities.append({"household_id": household_id, "entity_type": "phone", "canonical": p, "canonical_hash": h, "meta": {}})
    for o in ORGS:
        h = _hash_canonical(o)
        if (o, h) in seen:
            continue
        seen.add((o, h))
        entities.append({"household_id": household_id, "entity_type": "org", "canonical": o, "canonical_hash": h, "meta": {}})
    for m in MERCHANTS:
        h = _hash_canonical(m)
        if (m, h) in seen:
            continue
        seen.add((m, h))
        entities.append({"household_id": household_id, "entity_type": "merchant", "canonical": m, "canonical_hash": h, "meta": {}})
    for t in TOPICS:
        h = _hash_canonical(t)
        if (t, h) in seen:
            continue
        seen.add((t, h))
        entities.append({"household_id": household_id, "entity_type": "topic", "canonical": t, "canonical_hash": h, "meta": {}})
    for a in ACCOUNT_HASHES[:10]:
        entities.append({"household_id": household_id, "entity_type": "account", "canonical": a, "canonical_hash": a, "meta": {}})
    return entities


SESSION_SUMMARY_TEXTS = [
    "Conversation included reminders and a brief mention of a call.",
    "Routine check-in; discussed appointments and medication.",
    "Call touched on banking and a follow-up with the doctor.",
    "Short check-in; no concerns raised.",
    "Discussed bills and a planned family visit.",
    "Reminders set; caregiver to call back later.",
    "Talked about prescriptions and refill dates.",
    "General catch-up; one question about a letter.",
    "Brief call; confirmed weekend plans.",
    "Conversation included a question about insurance and a reminder to pay a bill.",
]


def generate_summaries(household_id: str, session_ids: list[str], days_back: int = 180) -> list[dict]:
    summaries = []
    # Session-scoped (varied text so list view is useful)
    for i, sid in enumerate(session_ids[:30]):
        summaries.append({
            "household_id": household_id,
            "session_id": sid,
            "period_start": None,
            "period_end": None,
            "summary_text": SESSION_SUMMARY_TEXTS[i % len(SESSION_SUMMARY_TEXTS)],
            "summary_json": {},
        })
    # Period-scoped (with signal_count so trend chart shows bars)
    for _ in range(20):
        end = datetime.now(timezone.utc) - timedelta(days=random.randint(0, days_back))
        start = end - timedelta(days=random.randint(1, 7))
        summaries.append({
            "household_id": household_id,
            "session_id": None,
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "summary_text": "Weekly summary: routine check-ins and one potential outreach call noted.",
            "summary_json": {"signal_count": random.randint(0, 5)},
        })
    return summaries


def generate_risk_signals(household_id: str, count: int = 80) -> list[dict]:
    types_ = ["relational_anomaly", "urgency_language", "sensitive_request", "unusual_payee", "account_takeover_risk", "motif_medicare", "motif_irs"]
    statuses = ["open", "acknowledged", "dismissed", "escalated"]
    signals = []
    base_ts = datetime.now(timezone.utc) - timedelta(days=90)
    for i in range(count):
        ts = (base_ts + timedelta(days=random.randint(0, 80), hours=random.randint(0, 23))).isoformat()
        severity = random.randint(1, 5)
        status = random.choices(statuses, weights=[40, 25, 25, 10])[0]
        signals.append({
            "household_id": household_id,
            "ts": ts,
            "signal_type": random.choice(types_),
            "severity": severity,
            "score": round(random.uniform(0.3, 0.95), 3),
            "explanation": {"motif": "urgency", "entities": []},
            "recommended_action": {"action": "review_with_elder", "priority": "medium"},
            "status": status,
        })
    return signals


def generate_watchlists(household_id: str, count: int = 15) -> list[dict]:
    wl = []
    for i in range(count):
        wl.append({
            "household_id": household_id,
            "watch_type": random.choice(["entity_hash", "keyword", "merchant"]),
            "pattern": {"keywords": ["Medicare", "IRS", "gift card"], "entity_hashes": []},
            "reason": "High-risk scam patterns",
            "priority": random.randint(0, 5),
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=90)).isoformat() if random.random() < 0.5 else None,
        })
    return wl


def generate_agent_runs(household_id: str, count: int = 40) -> list[dict]:
    runs = []
    base_ts = datetime.now(timezone.utc) - timedelta(days=60)
    for i in range(count):
        started = base_ts + timedelta(days=random.randint(0, 55), hours=random.randint(0, 23))
        ended = started + timedelta(minutes=random.randint(1, 10))
        status = random.choice(["completed", "completed", "failed"])
        runs.append({
            "household_id": household_id,
            "agent_name": "financial_security",
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "status": status,
            "summary_json": {"signals_created": random.randint(0, 5), "sessions_processed": random.randint(1, 20)},
            "step_trace": [
                {"step": "ingest", "status": "success"},
                {"step": "normalize", "status": "success"},
                {"step": "detect", "status": "success"},
                {"step": "persist", "status": "success"},
            ],
        })
    return runs


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Seed Supabase with thousands of data points for one user.")
    parser.add_argument("--dry-run", action="store_true", help="Only print counts, do not connect to DB")
    parser.add_argument("--household-id", type=str, default=None, help="Use existing household (with --user-id); only add data")
    parser.add_argument("--user-id", type=str, default=None, help="Use existing auth user (with --household-id) or for new seed household")
    parser.add_argument("--sessions", type=int, default=150, help="Number of sessions")
    parser.add_argument("--days-back", type=int, default=180, help="Spread sessions over this many days")
    parser.add_argument("--output-json", type=str, default=None, help="Also write generated events/sessions to this JSON file (no Supabase needed)")
    args = parser.parse_args()

    # Default to demo placeholder when set (config/demo_placeholder.json or DEMO_USER_ID / DEMO_HOUSEHOLD_ID)
    if not args.household_id and not args.user_id:
        try:
            from config.demo_placeholder import get_demo_placeholder
            ph = get_demo_placeholder()
            if ph:
                args.household_id = ph["household_id"]
                args.user_id = ph["user_id"]
        except Exception:
            pass
    use_existing_household = bool(args.household_id and args.user_id)

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not args.dry_run and (not url or not key):
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (e.g. in .env or apps/api/.env)", file=sys.stderr)
        sys.exit(1)

    # 1) Auth user and household
    seed_user_id = args.user_id or os.environ.get("SEED_USER_ID")
    use_existing_household = bool(args.household_id and args.user_id)
    if args.dry_run:
        seed_user_id = seed_user_id or str(uuid4())
        print("[dry-run] Would create or use auth user:", seed_user_id)
        if use_existing_household:
            print("[dry-run] Using existing household:", args.household_id)
    elif use_existing_household:
        seed_user_id = args.user_id
        household_id = args.household_id
        print("Using household (demo placeholder or --household-id/--user-id):", household_id, "user:", seed_user_id)
    elif not seed_user_id:
        try:
            from supabase import create_client
            client = create_client(url, key)
            # Create user via admin API (service role only)
            try:
                r = client.auth.admin.create_user({"email": SEED_EMAIL, "password": SEED_PASSWORD, "email_confirm": True})
                if r and getattr(r, "user", None):
                    seed_user_id = str(r.user.id)
                    print("Created auth user:", SEED_EMAIL, "id:", seed_user_id)
                else:
                    seed_user_id = _find_seed_user_by_email(client)
                    if not seed_user_id:
                        print("Could not create or find auth user. Set SEED_USER_ID=<uuid> and create user in Dashboard.", file=sys.stderr)
                        sys.exit(1)
            except Exception as e:
                seed_user_id = _find_seed_user_by_email(client)
                if not seed_user_id:
                    print("Auth create_user failed:", e, file=sys.stderr)
                    print("Create a user in Supabase Dashboard (Authentication → Add user) and run with SEED_USER_ID=<uuid>", file=sys.stderr)
                    sys.exit(1)
                print("Using existing auth user after create_user error:", seed_user_id)
        except Exception as e:
            print("Supabase auth error:", e, file=sys.stderr)
            sys.exit(1)
    else:
        print("Using SEED_USER_ID:", seed_user_id)

    if not use_existing_household:
        household_id = str(uuid4())
        device_ids = [str(uuid4()) for _ in range(3)]
    else:
        device_ids = [str(uuid4()) for _ in range(3)]  # add 3 devices to existing household

    # 2) Generate all in-memory
    print("Generating sessions...")
    sessions = generate_sessions(household_id, device_ids, num_sessions=args.sessions, days_back=args.days_back)
    print("Generating events...")
    all_events = []
    for i, sess in enumerate(sessions):
        evs = generate_events_for_session(
            "<session_id>",  # placeholder; we'll assign after insert
            sess["device_id"],
            sess["started_at"],
            sess["ended_at"],
            i,
        )
        for e in evs:
            e["_session_idx"] = i
        all_events.extend(evs)

    entities = generate_entities(household_id)
    summaries = generate_summaries(household_id, ["<sid>"] * 30, args.days_back)
    risk_signals = generate_risk_signals(household_id)
    watchlists = generate_watchlists(household_id)
    agent_runs = generate_agent_runs(household_id)

    if args.output_json:
        out = {
            "household_id": household_id,
            "device_ids": device_ids,
            "sessions": sessions,
            "events": all_events,
            "entities": entities,
            "summaries": summaries,
            "risk_signals": risk_signals,
            "watchlists": watchlists,
            "agent_runs": agent_runs,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("Wrote", len(all_events), "events and", len(sessions), "sessions to", args.output_json)

    if args.dry_run:
        print("[dry-run] household_id:", household_id)
        print("[dry-run] sessions:", len(sessions))
        print("[dry-run] events:", len(all_events))
        print("[dry-run] entities:", len(entities))
        print("[dry-run] summaries:", len(summaries))
        print("[dry-run] risk_signals:", len(risk_signals))
        print("[dry-run] watchlists:", len(watchlists))
        print("[dry-run] agent_runs:", len(agent_runs))
        return

    from supabase import create_client
    client = create_client(url, key)

    # 3) Insert: household, user (unless existing), devices
    now = datetime.now(timezone.utc).isoformat()
    if not use_existing_household:
        print("Inserting household, user, devices...")
        h_res = client.table("households").insert({"id": household_id, "name": "Seed Demo Household"}).execute()
        if not h_res.data:
            print("Failed to insert household", file=sys.stderr)
            sys.exit(1)
        client.table("users").insert({
            "id": seed_user_id,
            "household_id": household_id,
            "role": "caregiver",
            "display_name": "Seed User",
            "created_at": now,
            "updated_at": now,
        }).execute()
    else:
        print("Inserting devices (existing household)...")
    for did in device_ids:
        client.table("devices").insert({
            "id": did,
            "household_id": household_id,
            "device_type": "anchor_speaker",
            "firmware_version": "1.0.0",
            "last_seen_at": now,
        }).execute()

    # 4) Insert sessions (need real IDs)
    print("Inserting sessions...")
    BATCH = 100
    session_id_list = []
    for i in range(0, len(sessions), BATCH):
        chunk = sessions[i : i + BATCH]
        r = client.table("sessions").insert(chunk).execute()
        if r.data:
            session_id_list.extend([s["id"] for s in r.data])

    # Map session index -> session_id
    sid_by_idx = {}
    for i, sess in enumerate(sessions):
        if i < len(session_id_list):
            sid_by_idx[i] = session_id_list[i]

    # 5) Assign session_id to events and insert in batches
    for e in all_events:
        idx = e.pop("_session_idx", None)
        if idx is not None and idx in sid_by_idx:
            e["session_id"] = sid_by_idx[idx]
        elif not e.get("session_id") or str(e.get("session_id", "")).startswith("<"):
            # Fallback: first session
            e["session_id"] = session_id_list[0] if session_id_list else None
    session_ids_set = set(session_id_list)
    all_events = [e for e in all_events if e.get("session_id") in session_ids_set]

    print("Inserting events (batches of 500)...")
    event_id_by_key = {}  # (session_id, seq) -> event_id
    for i in range(0, len(all_events), 500):
        chunk = all_events[i : i + 500]
        # Remove any key not in schema
        rows = []
        for e in chunk:
            rows.append({
                "session_id": e["session_id"],
                "device_id": e["device_id"],
                "ts": e["ts"],
                "seq": e["seq"],
                "event_type": e["event_type"],
                "payload": e.get("payload", {}),
                "payload_version": e.get("payload_version", 1),
            })
        r = client.table("events").insert(rows).execute()
        if r.data:
            for ev in r.data:
                event_id_by_key[(ev["session_id"], ev["seq"])] = ev["id"]

    # 6) Entities (need IDs for mentions/relationships)
    print("Inserting entities...")
    entity_res = client.table("entities").insert(entities).execute()
    canonical_to_id = {}  # (entity_type, canonical_hash) -> uuid
    if entity_res.data:
        for row in entity_res.data:
            canonical_to_id[(row["entity_type"], row.get("canonical_hash") or _hash_canonical(row["canonical"]))] = row["id"]
    entity_ids = list(canonical_to_id.values()) if canonical_to_id else []

    # 7) Utterances from final_asr events
    print("Building and inserting utterances...")
    final_asr = [e for e in all_events if e.get("event_type") == "final_asr"]
    utterances = []
    for e in final_asr:
        payload = e.get("payload") or {}
        utterances.append({
            "session_id": e["session_id"],
            "ts": e["ts"],
            "speaker": (payload.get("speaker") or {}).get("role") or "unknown",
            "text": payload.get("text"),
            "text_hash": _hash_canonical(payload.get("text") or ""),
            "intent": None,
            "confidence": payload.get("confidence"),
        })
    utt_ids = []
    for i in range(0, len(utterances), 300):
        r = client.table("utterances").insert(utterances[i : i + 300]).execute()
        if r.data:
            utt_ids.extend([u["id"] for u in r.data])

    # 8) Mentions (session_id, utterance_id or event_id, entity_id)
    print("Inserting mentions...")
    mentions = []
    for i, u in enumerate(utterances[:len(utt_ids)]):
        if i >= len(utt_ids):
            break
        utt_id = utt_ids[i]
        sess_id = u["session_id"]
        ts = u["ts"]
        # Assign 0–3 random entities per utterance
        for _ in range(random.randint(0, 3)):
            if entity_ids:
                eid = random.choice(entity_ids)
                mentions.append({
                    "session_id": sess_id,
                    "utterance_id": utt_id,
                    "entity_id": eid,
                    "ts": ts,
                    "span": {"start": 0, "end": 5},
                    "confidence": round(random.uniform(0.6, 0.95), 2),
                })
    for i in range(0, len(mentions), 500):
        client.table("mentions").insert(mentions[i : i + 500]).execute()

    # 9) Relationships (CO_OCCURS between entity pairs)
    print("Inserting relationships...")
    rels = []
    seen_pairs = set()
    for _ in range(min(600, len(entity_ids) * 2)):
        if len(entity_ids) < 2:
            break
        a, b = random.sample(entity_ids, 2)
        if a == b:
            continue
        pair = (min(a, b), max(a, b))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        t = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 90))
        rels.append({
            "household_id": household_id,
            "src_entity_id": a,
            "dst_entity_id": b,
            "rel_type": "CO_OCCURS",
            "weight": round(random.uniform(0.5, 1.0), 2),
            "first_seen_at": t.isoformat(),
            "last_seen_at": (t + timedelta(days=random.randint(1, 30))).isoformat(),
            "evidence": [],
        })
    for i in range(0, len(rels), 200):
        client.table("relationships").insert(rels[i : i + 200]).execute()

    # 10) Summaries (fix session_id for session-scoped)
    print("Inserting summaries...")
    for i, s in enumerate(summaries):
        if s.get("session_id") == "<sid>" and i < len(session_id_list):
            s["session_id"] = session_id_list[i % len(session_id_list)]
        elif s.get("session_id") == "<sid>":
            s["session_id"] = session_id_list[0] if session_id_list else None
    client.table("summaries").insert(summaries).execute()

    # 11) Risk signals
    print("Inserting risk_signals...")
    for i in range(0, len(risk_signals), 100):
        client.table("risk_signals").insert(risk_signals[i : i + 100]).execute()

    # 12) Feedback (need risk_signal ids)
    rs_ids = []
    rq = client.table("risk_signals").select("id").eq("household_id", household_id).limit(50).execute()
    if rq.data:
        rs_ids = [r["id"] for r in rq.data]
    if rs_ids and seed_user_id:
        feedback_rows = []
        for rs_id in rs_ids[:25]:
            feedback_rows.append({
                "household_id": household_id,
                "risk_signal_id": rs_id,
                "user_id": seed_user_id,
                "label": random.choice(["true_positive", "false_positive", "unsure"]),
                "notes": "Seed feedback",
            })
        client.table("feedback").insert(feedback_rows).execute()
        print("Inserted", len(feedback_rows), "feedback rows")

    # 13) Watchlists
    client.table("watchlists").insert(watchlists).execute()

    # 14) Agent runs
    client.table("agent_runs").insert(agent_runs).execute()

    # 15) Device sync state
    for did in device_ids:
        client.table("device_sync_state").upsert({
            "device_id": did,
            "last_upload_ts": now,
            "last_upload_seq_by_session": {},
            "last_watchlist_pull_at": now,
            "updated_at": now,
        }, on_conflict="device_id").execute()

    print("Done.")
    print("  Household id:", household_id)
    if use_existing_household:
        print("  User id:", seed_user_id)
    else:
        print("  Auth user (login):", SEED_EMAIL, "/", SEED_PASSWORD)
    print("  Sessions:", len(session_id_list))
    print("  Events:", len(all_events))
    print("  Utterances:", len(utterances))
    print("  Entities:", len(entities))
    print("  Mentions:", len(mentions))
    print("  Relationships:", len(rels))
    print("  Risk signals:", len(risk_signals))
    print("  Watchlists:", len(watchlists))
    print("  Agent runs:", len(agent_runs))
    print("Run worker to run pipeline: python -m worker.main --household-id", household_id, "--once")


if __name__ == "__main__":
    main()
