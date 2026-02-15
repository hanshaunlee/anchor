#!/usr/bin/env python3
"""
One-time setup so "Notify caregiver" and "Action plan" work in the app.

Does three things:
  1. Enables outbound contact in household consent defaults.
  2. Sets consent on the latest session so the agents see it.
  3. Adds one caregiver contact (email) so notify has a recipient.

Usage (from repo root):

  # Use an existing user's household (get user UUID from Supabase → Authentication → Users)
  SEED_USER_ID=<auth-user-uuid> PYTHONPATH=apps/api:. python scripts/enable_alerts_setup.py

  # Or use a specific household (e.g. from GET /households/me)
  HOUSEHOLD_ID=<household-uuid> PYTHONPATH=apps/api:. python scripts/enable_alerts_setup.py

  # Optional: custom contact email (default: demo@example.com)
  CAREGIVER_EMAIL=you@example.com SEED_USER_ID=... PYTHONPATH=apps/api:. python scripts/enable_alerts_setup.py

Requires: .env or apps/api/.env with SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "apps" / "api"))


def _load_dotenv() -> None:
    for env_path in (ROOT / "apps" / "api" / ".env", ROOT / ".env"):
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


def main() -> None:
    _load_dotenv()
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env or apps/api/.env", file=sys.stderr)
        sys.exit(1)

    household_id = os.environ.get("HOUSEHOLD_ID")
    user_id = os.environ.get("SEED_USER_ID")

    # Try demo placeholder (e.g. from config/demo_placeholder.json)
    if not household_id and not user_id:
        try:
            from config.demo_placeholder import get_demo_placeholder
            ph = get_demo_placeholder()
            if ph:
                household_id = ph.get("household_id")
                user_id = ph.get("user_id")
        except Exception:
            pass

    from supabase import create_client
    client = create_client(url, key)
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()

    if not household_id and user_id:
        r = client.table("users").select("household_id").eq("id", user_id).limit(1).execute()
        if r.data and len(r.data) > 0:
            household_id = r.data[0]["household_id"]
        else:
            print("No user row for SEED_USER_ID. Sign in once in the app so POST /households/onboard runs, or run seed_supabase_data.py first.", file=sys.stderr)
            sys.exit(1)
    elif not household_id:
        r = client.table("users").select("id, household_id").limit(1).execute()
        if r.data and len(r.data) > 0:
            user_id = r.data[0]["id"]
            household_id = r.data[0]["household_id"]
        else:
            print("No users in DB. Sign up in the app (then run this again) or run seed_supabase_data.py first.", file=sys.stderr)
            sys.exit(1)

    print("Household:", household_id)
    if user_id:
        print("User:", user_id)

    # 1) Household consent defaults: allow outbound contact
    try:
        client.table("household_consent_defaults").upsert({
            "household_id": household_id,
            "allow_outbound_contact": True,
            "share_with_caregiver": True,
            "share_text": True,
            "escalation_threshold": 3,
            "updated_at": now,
        }, on_conflict="household_id").execute()
        print("  ✓ Household consent: allow_outbound_contact = true")
    except Exception as e:
        print("  ✗ household_consent_defaults upsert failed:", e, file=sys.stderr)
        sys.exit(1)

    # 2) Latest session: set consent_allow_outbound_contact so agents see it
    try:
        sess = (
            client.table("sessions")
            .select("id, consent_state")
            .eq("household_id", household_id)
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        )
        if sess.data and len(sess.data) > 0:
            sid = sess.data[0]["id"]
            consent = dict(sess.data[0].get("consent_state") or {})
            consent["consent_allow_outbound_contact"] = True
            client.table("sessions").update({"consent_state": consent}).eq("id", sid).execute()
            print("  ✓ Latest session consent: consent_allow_outbound_contact = true")
        else:
            print("  ⚠ No sessions for this household; create one by using the app or seeding. Consent defaults are set.")
    except Exception as e:
        print("  ✗ Session consent update failed:", e, file=sys.stderr)

    # 3) One caregiver contact (so notify has a recipient)
    email = os.environ.get("CAREGIVER_EMAIL", "demo@example.com")
    try:
        existing = (
            client.table("caregiver_contacts")
            .select("id")
            .eq("household_id", household_id)
            .limit(1)
            .execute()
        )
        if existing.data and len(existing.data) > 0:
            print("  ✓ Caregiver contact already exists")
        else:
            channels = {"email": {"email": email, "value": email}}
            row = {
                "household_id": household_id,
                "name": "Demo Caregiver",
                "relationship": "family",
                "channels": channels,
                "priority": 1,
                "quiet_hours": {},
                "verified": False,
                "created_at": now,
                "updated_at": now,
            }
            if user_id:
                row["user_id"] = user_id
            client.table("caregiver_contacts").insert(row).execute()
            print("  ✓ Added caregiver contact:", email)
    except Exception as e:
        print("  ✗ caregiver_contacts insert failed:", e, file=sys.stderr)
        print("  (Run migration 012_outbound_actions_caregiver_contacts.sql if needed)", file=sys.stderr)

    print()
    print("Done. Next:")
    print("  1. Open the app and sign in as a caregiver for this household.")
    print("  2. Go to Alerts and open any risk signal.")
    print("  3. Use «Notify caregiver»: click Preview message, then Confirm send.")
    print("  4. Use «Action plan»: click Run Incident Response, then complete tasks.")
    print()
    print("(Outbound messages use the mock provider by default; no real SMS/email sent.)")


if __name__ == "__main__":
    main()
