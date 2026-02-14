# Seeding Supabase with Demo Data

The script `scripts/seed_supabase_data.py` generates **a few thousand data points** for one user/household, matching the full Supabase schema and the Anchor product (elder/caregiver voice, financial security, scam detection).

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

```bash
# Use project venv
PYTHONPATH=apps/api:. .venv/bin/python3 scripts/seed_supabase_data.py
```

If your project does not allow creating auth users from the API (e.g. “Database error creating new user”):

1. In **Supabase Dashboard → Authentication → Users**, click **Add user** and create a user (e.g. `seed@anchor.demo` / any password).
2. Copy the user’s **UUID** from the list.
3. Run:
   ```bash
   SEED_USER_ID=<paste-uuid-here> PYTHONPATH=apps/api:. .venv/bin/python3 scripts/seed_supabase_data.py
   ```
   This creates a **new** household and links that auth user to it, then seeds all data.

To add data to an **existing** household (user already onboarded via the app):

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

## Local-only: keep a file, no Supabase

To only generate data and save it locally (no upload):

```bash
PYTHONPATH=apps/api:. .venv/bin/python3 scripts/seed_supabase_data.py --dry-run --output-json data/seed_export.json
```

This creates `data/seed_export.json` with sessions, events, entities, risk_signals, watchlists, and agent_runs in the same shapes the script would insert. You can use this for tests or a separate import path later.
