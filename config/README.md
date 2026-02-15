# Config

- **`demo_placeholder.json`** – Canonical demo user and household for this project. Replace the placeholder UUIDs with your real Supabase **user id** (Auth user UUID) and **household id**. Used by:
  - `scripts/seed_supabase_data.py` (default target when no `--household-id`/`--user-id`)
  - `scripts/load_elderly_conversations.py --demo`
  - `scripts/run_outreach_demo.py`, `run_financial_agent_demo.py`, `demo_replay.py`
  - Env override: `DEMO_USER_ID`, `DEMO_HOUSEHOLD_ID` (optional)
