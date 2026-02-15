# Seeding Supabase with Demo Data

The script `scripts/seed_supabase_data.py` generates **a few thousand data points** for one user/household, matching the full Supabase schema and the Anchor product (elder/caregiver voice, financial security, scam detection).

**Demo placeholder:** The canonical demo user and household for this project are set in **`config/demo_placeholder.json`** (or env `DEMO_USER_ID` / `DEMO_HOUSEHOLD_ID`). Replace the placeholder UUIDs with your real Supabase **user id** (Auth user UUID) and **household id**. All seed data, elderly conversation loads, and demo scripts then use this household and user.

## What gets created

| Table | Approx. count |
|------|----------------|
| households | 1 |
| users | 1 (linked to auth) |
| devices | 3 |
| sessions | 150 (spread over 180 days) |
| events | 2500–3000 (final_asr, intent, device_state, transaction_detected, payee_added, bank_alert_received) |
| utterances | ~1000+ (from final_asr) |
| entities | ~47 (person, phone, org, merchant, topic, account) |
| mentions | many (utterance ↔ entity) |
| relationships | ~600 (CO_OCCURS) |
| summaries | 50 (session- and period-scoped) |
| risk_signals | 80 (varied types, severity, status) |
| watchlists | 15 |
| feedback | up to 25 (on risk_signals) |
| agent_runs | 40 (financial_security) |
| device_sync_state | 3 |

Timestamps are spread over the last 180 days so time-range queries and the pipeline (e.g. `--time-window-days`) exercise the model.

## How to run

From repo root, with `.env` or `apps/api/.env` containing `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`:

**Recommended: use the demo placeholder**

1. Edit **`config/demo_placeholder.json`** and set `user_id` and `household_id` to your Supabase Auth user UUID and that user’s household UUID (from `public.users` / `public.households`).
2. Run:
   ```bash
   PYTHONPATH=apps/api:. .venv/bin/python3 scripts/seed_supabase_data.py
   ```
   This adds seed data (and optionally `--with-elderly-conversations`) to that household. No new user or household is created.

**Without demo placeholder (create new user/household)**

```bash
PYTHONPATH=apps/api:. .venv/bin/python3 scripts/seed_supabase_data.py
```

If your project does not allow creating auth users from the API, create a user in **Supabase Dashboard → Authentication → Users**, copy their UUID, then:

```bash
SEED_USER_ID=<paste-uuid-here> PYTHONPATH=apps/api:. .venv/bin/python3 scripts/seed_supabase_data.py
```
This creates a **new** household and links that auth user to it, then seeds all data.

**Explicit household and user (override placeholder)**

```bash
PYTHONPATH=apps/api:. .venv/bin/python3 scripts/seed_supabase_data.py --household-id <household-uuid> --user-id <auth-user-uuid>
```

## Options

- `--dry-run` – Print counts and exit; no DB writes.
- `--output-json <path>` – Write generated sessions/events/entities/risk_signals/etc. to a JSON file (useful for backup or local inspection). Does not require Supabase.
- `--sessions 200` – Number of sessions (default 150).
- `--days-back 90` – Spread sessions over this many days (default 180).
## After seeding

Run the pipeline for the new household so risk signals and watchlists are computed from the graph:

```bash
python -m worker.main --household-id <household_id> --once
```

(The script prints the `household_id` at the end.)

## Elderly conversation dataset (35k) — upload once to Supabase

To load **35,000 elderly conversations** (365 days, edge-upload schema) into the **demo household** (one-time):

1. Set **`config/demo_placeholder.json`** with your Supabase `user_id` and `household_id`.
2. Generate the dataset (if you don’t already have `data/elderly_conversations/`):
   ```bash
   python3 scripts/generate_elderly_conversations.py --num-sessions 35000 --days 365
   ```
3. Upload once to Supabase:
   ```bash
   PYTHONPATH=apps/api:. .venv/bin/python3 scripts/load_elderly_conversations.py --demo
   ```

To load into a **specific household** (no config):  
`PYTHONPATH=apps/api:. .venv/bin/python3 scripts/load_elderly_conversations.py --household-id <uuid>`

See `data/elderly_conversations/README.md` for schema.

## Demo mode (frontend toggle)

When you turn **demo mode ON** (bottom-left in the dashboard), the app uses static fixtures under `apps/web/public/fixtures/` instead of the API. Those fixtures are built from the 35k dataset so the UI shows **35,000 sessions** and proportional summaries/risk signals.

To refresh the fixtures after changing the elderly dataset:

```bash
python3 scripts/build_demo_fixtures.py
```

## Local-only: keep a file, no Supabase

To only generate data and save it locally (no upload):

```bash
PYTHONPATH=apps/api:. .venv/bin/python3 scripts/seed_supabase_data.py --dry-run --output-json data/seed_export.json
```

This creates `data/seed_export.json` with sessions, events, entities, risk_signals, watchlists, and agent_runs in the same shapes the script would insert. You can use this for tests or a separate import path later.
