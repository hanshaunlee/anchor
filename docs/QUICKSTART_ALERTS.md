# Quick start: Notify caregiver & Action plan

Get **Notify caregiver** and **Action plan** working with minimal steps.

---

## Do this first (fix 500s / dead buttons)

If the API returns 500 and the buttons do nothing, Supabase is missing tables. Do these **two steps in order**:

1. **Create the tables**  
   - Open [Supabase Dashboard](https://supabase.com/dashboard) → your project → **SQL Editor**.  
   - Open the file **`db/run_migrations_011_012_013.sql`** in this repo, **copy all** of it, paste into the SQL Editor, and click **Run**.  
   - Wait until it finishes (you may see “already exists” for some objects; that’s fine).

2. **Enable alerts for your household**  
   - In the same SQL Editor, open **`db/setup_alerts_one_household.sql`** (it’s already set for user `b1822f1d-...` and household `d38bd799-...`).  
   - Copy all, paste, and click **Run**.

3. **Reload the app**  
   - Refresh the alert page in the browser.  
   - “Preview message” and “Run Incident Response” should work (no more 500s from missing tables).  
   - If the web app can’t reach the API, ensure the API is running (`./scripts/run_api.sh`) and set `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` in `apps/web/.env.local`, then refresh again.

---

## Optional: performance & queue migrations

After 011–013, you can run these for better performance and the ingest→supervisor queue:

| Migration | Purpose |
|-----------|--------|
| **`db/migrations/016_rpc_alert_page_and_investigation_context.sql`** | RPCs to reduce round trips for alert page and investigation window (optional; alert page works without it). |
| **`db/migrations/017_performance_indexes.sql`** | Indexes for risk_signal_embeddings, outbound_actions, events, risk_signals. |
| **`db/migrations/018_processing_queue.sql`** | `processing_queue` table for enqueueing supervisor runs; worker can poll with `--poll`. |

- **Alert detail** now uses a single **`GET /risk_signals/{id}/page`** request (no waterfall). No extra setup required.
- **Processing queue**: run migrations **018** and **019** (dedupe, atomic claim, retry). Then either:
  - **Local worker**: start the worker with `--poll` (e.g. `python -m worker.main --poll --poll-interval 30` from `apps/worker` with `PYTHONPATH` including repo root and `apps/api`), or
  - **Modal**: set Modal secret `anchor-supabase` with `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`, then run **`modal run modal_queue.py`** to process one job (or deploy and add a schedule). The queue is the abstraction layer; you can switch between local poller and Modal.
- **Ingest**: after **`POST /ingest/events`**, the response includes **`pipeline_suggested: true`** when events were ingested; the UI or worker can then call **`POST /system/run_ingest_pipeline`** (or enqueue with `dedupe=true` to avoid storms).

---

## 1. Prerequisites

- Supabase project set up and **`db/bootstrap_supabase.sql`** run.
- **Run the alerts migrations** so the required tables exist. In Supabase Dashboard → **SQL Editor**, open **`db/run_migrations_011_012_013.sql`** from this repo, copy its full contents, paste into the editor, and click **Run**. This creates `household_capabilities`, `outbound_actions`, `caregiver_contacts`, `action_playbooks`, `action_tasks`, `incident_packets`, and related RLS. If you see "already exists" for a type or table, that’s fine; the script is idempotent where possible.
- `.env` (or `apps/api/.env`) with **SUPABASE_URL** and **SUPABASE_SERVICE_ROLE_KEY**.
- At least one user with a household (sign up in the app and complete onboarding, or use the seed script).

---

## 2. One-time setup (recommended)

Run the setup script once so consent and a caregiver contact exist:

```bash
# From repo root. Replace <auth-user-uuid> with your user's ID (Supabase → Authentication → Users).
SEED_USER_ID=<auth-user-uuid> PYTHONPATH=apps/api:. python scripts/enable_alerts_setup.py
```

**If you don’t know the user UUID:** sign in once in the app, then either:

- Use **HOUSEHOLD_ID** instead (copy from the network tab when you load the app, or from `GET /households/me`):

  ```bash
  HOUSEHOLD_ID=<household-uuid> PYTHONPATH=apps/api:. python scripts/enable_alerts_setup.py
  ```

- Or run the full seed script first (it creates a user and household), then run `enable_alerts_setup.py` with that **SEED_USER_ID**.

**Optional:** use your own email for the demo contact:

```bash
CAREGIVER_EMAIL=you@example.com SEED_USER_ID=<uuid> PYTHONPATH=apps/api:. python scripts/enable_alerts_setup.py
```

You should see something like:

- `✓ Household consent: allow_outbound_contact = true`
- `✓ Latest session consent: consent_allow_outbound_contact = true`
- `✓ Added caregiver contact: demo@example.com` (or your email)

---

## 3. Use the app

1. **Start API and web** (if not already running):

   ```bash
   # Terminal 1
   cd apps/api && uvicorn api.main:app --reload

   # Terminal 2
   cd apps/web && npm run dev
   ```

2. **Sign in** as a **caregiver** (or admin) for the household you set up.

3. **Open Alerts** and click any risk signal (or create one via the Financial agent / seed data).

4. **Notify caregiver**
   - Click **Preview message** → you see the caregiver and elder-safe message.
   - Click **Confirm send** → an outbound action is created (mock provider by default; no real SMS/email unless you configure one).

5. **Action plan**
   - Click **Run Incident Response**.
   - After it finishes, the playbook and tasks appear (call bank, notify caregiver, etc.).
   - Use **Mark done** on tasks as you complete them.

---

## 4. If you didn’t run the script (manual setup)

You need:

- **Consent:** outbound contact allowed for the household or the latest session.
- **Contact:** at least one row in `caregiver_contacts` for the household.

**Option A – API (with your JWT):**

```bash
# Allow outbound contact (caregiver only)
curl -X PATCH "http://localhost:8000/households/me/consent" \
  -H "Authorization: Bearer YOUR_JWT" \
  -H "Content-Type: application/json" \
  -d '{"allow_outbound_contact": true}'

# Add a caregiver contact (caregiver only)
curl -X POST "http://localhost:8000/households/me/contacts" \
  -H "Authorization: Bearer YOUR_JWT" \
  -H "Content-Type: application/json" \
  -d '{"name": "Me", "email": "you@example.com"}'
```

**Option B – Supabase SQL (replace `YOUR_HOUSEHOLD_ID`):**

```sql
-- Consent defaults
INSERT INTO household_consent_defaults (household_id, allow_outbound_contact, updated_at)
VALUES ('YOUR_HOUSEHOLD_ID', true, now())
ON CONFLICT (household_id) DO UPDATE SET allow_outbound_contact = true, updated_at = now();

-- One contact (channels shape expected by the agent)
INSERT INTO caregiver_contacts (household_id, name, relationship, channels, priority, verified, created_at, updated_at)
VALUES (
  'YOUR_HOUSEHOLD_ID',
  'Demo Caregiver',
  'family',
  '{"email": {"email": "demo@example.com", "value": "demo@example.com"}}',
  1,
  false,
  now(),
  now()
);
```

Then set the **latest session** consent so the agents see it:

```sql
UPDATE sessions
SET consent_state = COALESCE(consent_state, '{}') || '{"consent_allow_outbound_contact": true}'
WHERE household_id = 'YOUR_HOUSEHOLD_ID'
  AND id = (
    SELECT id FROM sessions
    WHERE household_id = 'YOUR_HOUSEHOLD_ID'
    ORDER BY started_at DESC LIMIT 1
  );
```

---

## 5. Troubleshooting

| Issue | What to do |
|--------|------------|
| **Notify: "Consent does not allow outbound contact"** | Run `enable_alerts_setup.py` or set `allow_outbound_contact` / `consent_allow_outbound_contact` as above. |
| **Notify: no recipient / always demo@example.com** | Add a caregiver contact (script or `POST /households/me/contacts` with `name` and `email` or `phone`). |
| **Action plan: "Run Incident Response" does nothing** | Check browser network tab for 4xx/5xx. Ensure migrations **013** (action_playbooks, action_tasks, incident_packets) are applied. |
| **404 Not onboarded** | Sign in and complete onboarding (`POST /households/onboard`), or seed a user/household. |
| **403 Only caregivers or admins** | Use an account with role `caregiver` or `admin` for that household. |
| **500 / "Could not find the table … in the schema cache"** | Run **`db/run_migrations_011_012_013.sql`** in Supabase SQL Editor (see Prerequisites above). |

---

## Summary

1. Run **`scripts/enable_alerts_setup.py`** once with **SEED_USER_ID** or **HOUSEHOLD_ID**.
2. Sign in as a **caregiver**, open **Alerts** → one alert.
3. Use **Notify caregiver** (Preview → Confirm) and **Run Incident Response** for the action plan.
