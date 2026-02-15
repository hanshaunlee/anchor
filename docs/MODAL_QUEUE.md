# When Modal Runs (Heavy Work Offload)

Modal is used to run **intense work** (ingest pipeline, narrative agent, financial agent) so it doesn’t block your API or run slowly on your machine.

---

## What is the queue?

The **queue** is just a table in your database (`processing_queue`) that holds **jobs** like “run the investigation pipeline for this household.”

- **Enqueue** = add a row: “please run this job when you can.”
- **Process the queue** = something (a **worker**) repeatedly looks for pending rows, takes one, runs the heavy work, then marks it done.

So instead of “run this heavy thing right now on the API server,” you say “put it in the queue” and return quickly. A **worker** (your local poller or **Modal** on a schedule) picks up jobs and runs them. That way:

- The API stays fast and doesn’t time out.
- Heavy work (investigation = financial + narrative agents, LLM, etc.) runs on **Modal** (or a separate worker), not on your laptop or API server.

**Investigation** is exactly that heavy work. When you “Run Investigation” in the UI, it’s the same pipeline (INGEST_PIPELINE: financial agent, narrative agent, outreach). So:

- **Run inline** (default): API runs it now and waits → can be slow, risk of timeouts.
- **Run via queue** (`enqueue=true`): API adds a job to the queue and returns a `job_id` right away. Modal (or local worker) picks it up within ~2 minutes and runs the investigation on Modal. UI can show “Investigation running in background…” and refresh when done.

So yes, **investigation should (and can) run on Modal** when you use the queue: use **Run Investigation** with `enqueue=true`, and have Modal (or the worker) deployed so it processes the queue.

---

## When does Modal kick in?

1. **You enqueue work**  
   - From the API: `POST /system/run_ingest_pipeline` with `enqueue=true`, or  
   - After ingest: response has `pipeline_suggested: true` and your app/worker enqueues a run.  
   - Jobs go into `processing_queue` in Supabase.

2. **Something must process the queue**  
   - **Option A – Modal (recommended for heavy work):**  
     - **Database:** Run migrations 018 and 019 on Supabase (`processing_queue` and `rpc_claim_processing_queue_job`). Otherwise you get "Could not find the function public.rpc_claim_processing_queue_job".  
     - Create the secret (if you see “Secret 'anchor-supabase' not found”, create it first):  
       `modal secret create anchor-supabase SUPABASE_URL=<url> SUPABASE_SERVICE_ROLE_KEY=<key>`  
       Or use the link Modal shows in the error (e.g. Modal dashboard → Secrets → create `anchor-supabase` with those two keys).  
     - Deploy:  
       `modal deploy modal_queue.py`  
     - After deploy, Modal runs **every 2 minutes** and processes one pending job per run (ingest pipeline = financial + narrative + outreach). So Modal “kicks in” automatically; no need to run anything on your device.  
   - **Option B – Local worker:**  
     - Run:  
       `python -m worker.main --poll --poll-interval 30`  
     - (from `apps/worker` with `PYTHONPATH` including repo root and `apps/api`.)  
     - Good for dev; for production, Modal avoids overloading your API server.

3. **One-off run (no schedule)**  
   - `modal run modal_queue.py`  
   - Processes **one** queue job and exits. Use for testing or manual runs.

## What actually runs on Modal?

When Modal processes a `run_supervisor_ingest` job, it runs the full **supervisor INGEST_PIPELINE**, which includes:

- Ingest / normalize  
- **Financial security agent** (risk scoring, GNN if configured)  
- **Evidence narrative agent** (LLM-backed narratives, batched)  
- Outreach candidates / watchlist  

So the slow, intense calculations (agents, LLM, graph) run on Modal; your API stays responsive and returns quickly (e.g. with `job_id` when using enqueue).

## Summary

| You do | Modal does |
|--------|------------|
| `modal deploy modal_queue.py` | Every 2 min: claim one pending job and run the pipeline (agents, narrative, etc.) on Modal. |
| `modal run modal_queue.py` | Run once: process one job and exit. |
| Enqueue via API (e.g. `POST /system/run_ingest_pipeline` with `enqueue=true`) | Job sits in `processing_queue` until the next Modal run (or local worker poll) picks it up. |

So Modal is “used” as soon as you **deploy** the queue runner; it then runs on a schedule. For one-off tests, use `modal run modal_queue.py`.
