# Connecting Supabase to Anchor

Anchor uses Supabase for **auth** (web login), **database** (API + worker), and **JWT verification**. This doc covers **keys and env**. For **full setup** (create project, run schema bootstrap, link first user / onboard): see **[SUPABASE_SETUP.md](SUPABASE_SETUP.md)**.

---

## 1. Create a Supabase project

1. Go to [supabase.com](https://supabase.com) and sign in.
2. **New project** → choose org, name, database password, region.
3. Wait for the project to finish provisioning.

---

## 2. Get your API keys

1. In the Supabase Dashboard, open **Project Settings** (gear) → **API**.
2. Copy:
   - **Project URL** → use for `SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_URL`
   - **anon public** key → use for `SUPABASE_ANON_KEY` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - **service_role** key → use for `SUPABASE_SERVICE_ROLE_KEY` (server-only; never expose in the browser)

---

## 3. Set environment variables

### Option A: Single `.env` at repo root (API + worker)

Create a `.env` in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and set:

- `SUPABASE_URL` = your Project URL  
- `SUPABASE_ANON_KEY` = anon public key  
- `SUPABASE_SERVICE_ROLE_KEY` = service_role key  

The **API** (`apps/api`) and **worker** (`apps/worker`) read these when run from the root.

### Option B: Web app (Next.js)

The web app needs the **public** Supabase URL and anon key so the browser can talk to Supabase Auth.

Create **`apps/web/.env.local`** (or set these in your root `.env` if Next is configured to load it):

```env
NEXT_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here
```

Restart the Next dev server after changing env vars.

---

## 4. Database schema (if you use API/worker)

The API and worker expect tables such as: `users`, `households`, `sessions`, `events`, `risk_signals`, `risk_signal_embeddings`, `feedback`, `household_calibration`, `watchlists`, `summaries`, `devices`, `device_sync_state`. Run the SQL migrations in **`db/migrations/`** in order (e.g. in the Supabase Dashboard → SQL Editor): `001_initial_schema.sql`, `002_rls.sql`, `003_risk_signal_embeddings.sql`, `004_rls_embeddings_calibration.sql`, `005_agent_runs.sql`, `006_agent_runs_step_trace.sql`, `007_risk_signal_embeddings_extended.sql`. The last adds `dim`, `model_name`, `has_embedding`, etc., for GNN similar-incidents and embedding-centroid watchlists.

---

## 5. Verify

- **Web**: Run `npm run dev` in `apps/web`. Open the login page; if Supabase is connected, sign-in will hit Supabase Auth (no “Supabase not configured” in the UI).
- **API**: Run the API and call a protected endpoint with a valid Supabase JWT. If Supabase is misconfigured, you’ll get `503 Supabase not configured`.
- **Worker**: The worker uses `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` for ingest and pipelines; if unset, it logs a warning and uses placeholder data.

---

## Summary

| Variable                         | Where used        | Purpose                    |
|----------------------------------|-------------------|----------------------------|
| `SUPABASE_URL`                   | API, worker       | Server Supabase client     |
| `SUPABASE_SERVICE_ROLE_KEY`      | API, worker       | Server Supabase client     |
| `NEXT_PUBLIC_SUPABASE_URL`       | Web (Next.js)     | Browser Supabase client    |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY`  | Web (Next.js)     | Browser Supabase client    |

Keep the **service_role** key only in server env (e.g. `.env` at root or API/worker env); never put it in `NEXT_PUBLIC_*` or commit it.
