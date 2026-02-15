# Supabase setup for Anchor

Use this guide to set up a **new** Supabase project so it matches the Anchor schema and works with the API, worker, and web app.

**Full stack checklist:** For a single place that covers Supabase, env vars, auth, running API/web/worker, and verification, see **[SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)**.

**Notify caregiver & Action plan:** To turn on alerts and the action plan in the app with one script, see **[QUICKSTART_ALERTS.md](QUICKSTART_ALERTS.md)**.

## 1. Create a Supabase project

1. Go to [supabase.com/dashboard](https://supabase.com/dashboard) and sign in.
2. **New project** → choose org, name, database password (save it), region.
3. Wait for the project to be ready.

## 2. Run the full schema (bootstrap)

1. In the dashboard, open **SQL Editor**.
2. Open the file **`db/bootstrap_supabase.sql`** from this repo and copy its entire contents.
3. Paste into the SQL Editor and click **Run**.

This creates all tables, enums, indexes, RLS policies, `agent_runs.step_trace`, and the extended **`risk_signal_embeddings`** columns (`dim`, `model_name`, `has_embedding`, etc.) used for similar-incidents and embedding-centroid watchlists. **Run it only once** on an empty database (re-running will error on existing types/policies). If you already ran an older bootstrap, run **`db/migrations/007_risk_signal_embeddings_extended.sql`** in SQL Editor to add the new columns.

## 3. Configure your app (.env)

In the project root `.env` (or `apps/api/.env`), set:

```env
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service_role key>
```

Get both from **Project Settings → API**:  
- **Project URL** → `SUPABASE_URL`  
- **service_role** (secret) → `SUPABASE_SERVICE_ROLE_KEY`

Optional (for running migrations from your machine with `scripts/run_migration.py`):

```env
DATABASE_URL=postgresql://postgres.[ref]:[YOUR-PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres
```

Use the **Connection string** from **Project Settings → Database** (Connection pooling). Encode special characters in the password (`#` → `%23`, `%` → `%25`).

## 4. Auth (for API and web app)

- **Authentication → Providers**: enable **Email** (or others you need).
- The API uses **Bearer JWT** from Supabase Auth; the web app should sign in with Supabase Auth and send the access token.

## 5. Link the first user to a household

The API expects each authenticated user to have a row in `public.users` with `id = auth.uid()` and a `household_id`. RLS uses `auth.user_household_id()` (which reads from `users`).

**Option C – Web app sign-up + onboard (recommended)**  
The frontend supports **Create account** (sign-up). After Supabase Auth sign-up, the app calls `POST /households/onboard` with the user’s JWT; the API creates a new household and a `users` row (role `caregiver`). No trigger or manual seed needed. If the user confirms email later, they can go to **Sign in** and will be redirected to **Set up your household** (`/onboard`) once if they don’t yet have a `users` row.

**Option A – After first sign-up (manual seed)**  
1. Create a user via your app or **Authentication → Users → Add user** in the dashboard.  
2. Copy that user’s **UUID** (e.g. `a1b2c3d4-...`).  
3. In **SQL Editor**, run (replace `USER_UUID` and optionally the household name):

```sql
INSERT INTO households (id, name)
VALUES (gen_random_uuid(), 'My Household');

INSERT INTO users (id, household_id, role, display_name)
VALUES (
  'USER_UUID',
  (SELECT id FROM households ORDER BY created_at DESC LIMIT 1),
  'caregiver',
  'First User'
);
```

**Option B – Auto-seed on sign-up (trigger)**  
You can add a trigger on `auth.users` that inserts a new household and a `users` row when someone signs up. Example:

```sql
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
DECLARE
  new_household_id UUID;
BEGIN
  INSERT INTO households (name) VALUES ('Household') RETURNING id INTO new_household_id;
  INSERT INTO users (id, household_id, role, display_name)
  VALUES (NEW.id, new_household_id, 'caregiver', COALESCE(NEW.raw_user_meta_data->>'display_name', 'User'));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
```

Then every new Auth user gets a household and a `users` row.

## 6. Troubleshooting: 500 on signup (`unexpected_failure`)

If **Create account** returns a 500 and Supabase logs show `x_sb_error_code: unexpected_failure` on `/auth/v1/signup`, the failure is in **Supabase Auth**, not the Anchor API. Common causes:

1. **Trigger on `auth.users`**  
   **Quick fix:** Run **`db/drop_signup_trigger.sql`** in SQL Editor. That removes the trigger so signup no longer runs it; the frontend will create the household and user via `POST /households/onboard` (Option C) after signup. Ensure **`db/repair_households_users.sql`** has been run first so those tables exist for the API.  
   To inspect the error instead: in **Log Explorer** (Dashboard → Logs), run:
   ```sql
   select cast(postgres_logs.timestamp as datetime) as timestamp, event_message,
     parsed.error_severity, parsed.detail, parsed.sql_state_code
   from postgres_logs
   cross join unnest(metadata) as metadata
   cross join unnest(metadata.parsed) as parsed
   where regexp_contains(parsed.error_severity, 'ERROR|FATAL|PANIC')
     and regexp_contains(parsed.user_name, 'supabase_auth_admin')
   order by timestamp desc limit 20;
   ```
   Fix or drop the trigger (see [Resolving 500 status authentication errors](https://supabase.com/docs/guides/troubleshooting/resolving-500-status-authentication-errors-7bU5U8)).

2. **Email / SMTP**  
   If **Confirm email** is enabled and sending the confirmation email fails (e.g. SMTP misconfigured), Auth can return 500. For quick testing, turn off **Authentication → Providers → Email → Confirm email**.

3. **Auth schema / constraints**  
   Avoid foreign keys *from* `auth` schema to your tables, and avoid modifying Auth tables. Use **Option C** (no trigger): signup then `POST /households/onboard`.

4. **`relation "households" does not exist` (sql_state_code 42P01)**  
   Your trigger (Option B) runs on signup and inserts into `households` and `users`, but those tables don’t exist yet. **If you already ran a similar bootstrap before** (and don’t want to re-run the full file and hit “already exists” errors), run **`db/repair_households_users.sql`** in SQL Editor instead—it only creates the enum, `households`, and `users` if missing, and is safe to run multiple times. If you never ran the bootstrap, run the full **`db/bootstrap_supabase.sql`** once.

## 7. Verify

- **API**: Start the API, call an endpoint that uses `require_user` (e.g. `GET /agents/status`) with a valid Supabase JWT in `Authorization: Bearer <token>`.
- **Agents**: Run the financial agent (e.g. `POST /agents/financial/run` with `dry_run: true`). After a real run, check **Table Editor → agent_runs** for a row with `step_trace`.

## Summary

| Step | Action |
|------|--------|
| 1 | Create Supabase project |
| 2 | Run **`db/bootstrap_supabase.sql`** in SQL Editor (once) |
| 3 | Set **SUPABASE_URL** and **SUPABASE_SERVICE_ROLE_KEY** in `.env` |
| 4 | Enable Auth providers (e.g. Email) |
| 5 | Seed first user: add household + `users` row (manual or trigger) |
| 6 | If signup returns 500: see **Troubleshooting** (triggers, email, Auth schema) |
| 7 | Test API with JWT and check `agent_runs` / app flows |

This matches the schema and RLS used by the Anchor API, worker, and agents (including `agent_runs` and `step_trace`).
