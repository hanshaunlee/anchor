"""
Modal job runner for processing_queue. Heavy work (ingest pipeline, narrative/financial agents)
runs on Modal instead of on your API server or device.

When Modal kicks in:
- Deploy with schedule: modal deploy modal_queue.py  → run_one_job runs every 2 min and processes
  any pending queue jobs (ingest pipeline, which includes narrative + financial agents).
- Or run once: modal run modal_queue.py  → process one job and exit.

Create secret first: modal secret create anchor-supabase SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=...
"""
from __future__ import annotations

import os
import sys

try:
    import modal
except ImportError:
    modal = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

if modal is not None:
    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("supabase", "httpx", "python-dotenv")
        .add_local_dir(
            _REPO_ROOT,
            remote_path="/root/anchor",
            copy=True,
            ignore=[
                ".venv",
                ".git",
                "__pycache__",
                "*.pyc",
                ".next",
                "**/.next",
                "**/.next/**",
                "node_modules",
                "**/node_modules/**",
                "runs",
                ".env",
                ".env.*",
            ],
        )
    )
    _app = modal.App("anchor-queue", image=_image)
    # Create in Modal: modal secret create anchor-supabase SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=...
    _secret = modal.Secret.from_name("anchor-supabase")

    @_app.function(
        timeout=1800,
        secrets=[_secret],
        schedule=modal.Cron("*/2 * * * *"),  # Every 2 min when deployed: Modal kicks in automatically
    )
    def run_one_job() -> dict:
        """
        Claim one pending job from processing_queue (via RPC), run it, update row.
        Returns result_summary for the job runner interface.
        """
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            return {"ok": False, "reason": "missing_supabase_secret", "processed": False}

        sys.path.insert(0, "/root/anchor")
        sys.path.insert(0, "/root/anchor/apps/api")
        sys.path.insert(0, "/root/anchor/apps/worker")

        from datetime import datetime, timezone
        from supabase import create_client

        supabase = create_client(url, key)

        # Atomic claim (requires db migrations 018 + 019: processing_queue + rpc_claim_processing_queue_job)
        try:
            claimed = supabase.rpc("rpc_claim_processing_queue_job").execute()
        except Exception as e:
            err = str(e)
            if "rpc_claim_processing_queue_job" in err or "PGRST202" in err:
                return {
                    "ok": False,
                    "reason": "missing_rpc",
                    "processed": False,
                    "hint": "Run db migrations 018 and 019 on Supabase (processing_queue + rpc_claim_processing_queue_job).",
                    "error": err[:500],
                }
            return {"ok": False, "reason": "claim_failed", "processed": False, "error": err[:500]}

        if not claimed.data or len(claimed.data) == 0:
            return {"ok": True, "processed": False, "reason": "no_pending_job"}

        row = claimed.data[0]
        job_id = row["id"]
        household_id = str(row["household_id"])
        job_type = row.get("job_type") or "run_supervisor_ingest"
        payload = row.get("payload") or {}
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            if job_type == "run_supervisor_ingest":
                from worker.worker.jobs import run_supervisor_ingest_pipeline
                result = run_supervisor_ingest_pipeline(
                    supabase,
                    household_id,
                    time_range_start=None,
                    time_range_end=None,
                    dry_run=False,
                )
                result_summary = {
                    "supervisor_run_id": result.get("supervisor_run_id"),
                    "mode": result.get("mode"),
                    "created_signal_ids": result.get("created_signal_ids", []),
                }
            else:
                raise ValueError(f"Unknown job_type: {job_type}")

            supabase.table("processing_queue").update({
                "status": "completed",
                "completed_at": now_iso,
                "last_error": None,
                "next_attempt_at": None,
            }).eq("id", job_id).execute()

            return {"ok": True, "processed": True, "job_id": job_id, "result_summary": result_summary}
        except Exception as e:
            import logging
            logging.exception("Modal queue job %s failed: %s", job_id, e)
            err_text = str(e)[:1000]
            attempt = int(row.get("attempt_count") or 1)
            max_attempts = 3
            if attempt < max_attempts:
                from datetime import timedelta
                backoff_mins = [5, 15, 60][min(attempt - 1, 2)]
                next_at = (datetime.now(timezone.utc) + timedelta(minutes=backoff_mins)).isoformat()
                supabase.table("processing_queue").update({
                    "status": "pending",
                    "next_attempt_at": next_at,
                    "last_error": err_text,
                }).eq("id", job_id).execute()
            else:
                supabase.table("processing_queue").update({
                    "status": "failed",
                    "completed_at": now_iso,
                    "last_error": err_text,
                    "error_text": err_text,
                }).eq("id", job_id).execute()
            return {"ok": False, "processed": True, "job_id": job_id, "error": err_text}

    @_app.local_entrypoint()
    def main() -> None:
        """Run one queue job. Usage: modal run modal_queue.py"""
        out = run_one_job.remote()
        print(out)
