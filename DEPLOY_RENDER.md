# Deploy Anchor on Render

## 1. API (FastAPI)

### One-time: Connect repo and create Web Service

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**.
2. Connect your GitHub account and select the **Anchor** repo.
3. Render can auto-detect `render.yaml`. If it does, it will create **anchor-api** from the blueprint. Otherwise configure manually:
   - **Name:** `anchor-api` (or any name).
   - **Region:** Oregon (or your choice).
   - **Branch:** `main`.
   - **Runtime:** Python 3.
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `PYTHONPATH=.:apps/api uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Health check path:** `/health` (optional but recommended).

### Required environment variables (API)

Set these in the service **Environment** tab:

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL (Project Settings → API) |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service_role key (keep secret) |

Without these, the API returns 503 Supabase not configured.

### Optional environment variables

- `DATABASE_URL` — for migrations (if you run them from elsewhere).
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — Neo4j for graph view sync.
- `ANTHROPIC_API_KEY` — for Explain API plain-language descriptions.
- `OPENAI_API_KEY` — optional Evidence Narrative LLM.
- `ANCHOR_ML_CONFIG`, `ANCHOR_ML_CHECKPOINT_PATH` — GNN scoring (omit for rule-only mode).

### Deploy

- **Blueprint:** After connecting the repo, use **Apply** or **New** → **Blueprint** and point to this repo; Render will create the web service from `render.yaml`.
- **Manual:** After saving the Web Service, Render will build and deploy. Each push to `main` will auto-deploy if **Auto-Deploy** is on.

Your API URL will be like `https://anchor-api-xxxx.onrender.com`. Use it as `NEXT_PUBLIC_API_BASE_URL` for the web app.

---

## 2. Web dashboard (Next.js) — optional

To run the Next.js dashboard on Render:

1. **New** → **Web Service**.
2. Connect the same repo; set **Root Directory** to `apps/web`.
3. **Runtime:** Node.
4. **Build command:** `npm install && npm run build`
5. **Start command:** `npm start` (or `npx next start`).
6. **Environment:**  
   - `NEXT_PUBLIC_API_BASE_URL` = your Anchor API URL (e.g. `https://anchor-api-xxxx.onrender.com`)  
   - `NEXT_PUBLIC_SUPABASE_URL` = your Supabase URL  
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY` = your Supabase anon key  

---

## 3. Worker (background jobs)

The worker (`apps/worker`) runs pipeline jobs (e.g. ingest, risk scoring). On Render you can run it as a **Background Worker**:

- **Build:** same as API from repo root: `pip install -r requirements.txt`
- **Start:** `PYTHONPATH=.:apps:apps/api python -m worker.main --poll` (polls `processing_queue`; or use `--once --household-id <uuid>` for a one-off run).
- Set the same env vars as the API (Supabase, optional ML, etc.).

Cron-style scheduling is usually done with Render cron jobs or an external scheduler calling your API/worker endpoints.
