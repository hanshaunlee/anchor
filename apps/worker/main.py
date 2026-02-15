#!/usr/bin/env python3
"""
Worker entrypoint: run pipeline for a household or listen for jobs.
Usage: python -m worker.main [--household-id UUID] [--once]
"""
import argparse
import logging
import os
import sys

# Add repo root and apps/api for imports
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "apps", "api"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=str, default=None)
    parser.add_argument("--once", action="store_true", help="Run pipeline once then exit")
    parser.add_argument("--poll", action="store_true", help="Poll processing_queue every N seconds and run jobs")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between queue polls (default 30)")
    args = parser.parse_args()

    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            logger.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set; pipeline will use placeholder data")
        supabase = create_client(url, key) if url and key else None
    except Exception as e:
        logger.warning("Supabase client not available: %s", e)
        supabase = None

    if args.once and args.household_id:
        from worker.worker.jobs import run_pipeline
        result = run_pipeline(supabase, args.household_id)
        logger.info("Pipeline result: %s", list(result.keys()))
    elif args.poll and supabase:
        from worker.worker.jobs import process_one_processing_queue_job
        import time
        logger.info("Polling processing_queue every %s seconds", args.poll_interval)
        while True:
            if process_one_processing_queue_job(supabase):
                logger.info("Processed one queue job")
            time.sleep(args.poll_interval)
    else:
        logger.info("Worker idle (use --household-id and --once to run once, or --poll to poll processing_queue)")


if __name__ == "__main__":
    main()
