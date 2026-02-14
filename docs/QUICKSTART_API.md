# Quick start: API + frontend

The dashboard (Next.js on port 3000) calls the **Anchor API** (FastAPI on port 8000). If the API isn’t running, requests to `/households/me`, `/risk_signals`, etc. fail and you’ll see no response in DevTools.

## Run both

**Terminal 1 – API (required for real data)**  
From repo root:

```bash
./scripts/run_api.sh
```

You should see something like: `Uvicorn running on http://0.0.0.0:8000`. Leave this running.

**Terminal 2 – Frontend**  
From repo root or `apps/web` (if no root package.json, use `cd apps/web` first):

```bash
cd apps/web && npm run dev
# or from apps/web: npm run dev
```

Open http://localhost:3000 and log in. The dashboard will load data from the API.

## Check it

- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  
- In the app, after login you should see household info and risk signals (or empty lists if you haven’t seeded data yet).

## If you only want the UI (no backend)

Turn on **Demo mode** in the app. It uses local fixtures instead of the API so you don’t need the API running.
