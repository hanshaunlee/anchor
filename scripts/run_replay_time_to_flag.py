#!/usr/bin/env python3
"""
Replay events (real or synthetic) and compute time_to_flag.
Real: --events-file path/to/events.json  OR  --household-id ID [--time-window-days N] (Supabase).
Synthetic: --scenario scam|baseline (default when no real source given).
Output: time_to_flag (seconds from first event to first score above threshold), plus metrics JSON.
Usage (from repo root):
  PYTHONPATH=apps/api:. python scripts/run_replay_time_to_flag.py --events-file exports/events.json
  PYTHONPATH=apps/api:. python scripts/run_replay_time_to_flag.py --household-id hh1 --time-window-days 7
  PYTHONPATH=apps/api:. python scripts/run_replay_time_to_flag.py --scenario scam
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow importing api.pipeline when run from repo root
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(_repo_root / "apps" / "api"))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _load_dotenv() -> None:
    """Load .env from repo root so SUPABASE_* etc. are set when running from CLI."""
    env_file = _repo_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if "#" in v:
            v = v[: v.index("#")].strip()
        v = v.strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


def _ts_to_float(ev: dict) -> float:
    from datetime import datetime, timezone
    t = ev.get("ts")
    if t is None:
        return 0.0
    if isinstance(t, (int, float)):
        return float(t)
    if hasattr(t, "timestamp"):
        return t.timestamp()
    try:
        dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def load_events_from_file(path: Path) -> tuple[list[dict], str, str | None]:
    """Load events from JSON file. Returns (events, source_label, household_id_or_none).
    Accepts: [ {...}, ... ]  or  { "events": [...], "household_id": "..." }
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw, str(path), None
    if isinstance(raw, dict) and "events" in raw:
        events = raw["events"]
        hh = raw.get("household_id")
        label = f"file:{path}" + (f" (household={hh})" if hh else "")
        return events, label, hh
    raise ValueError(f"Expected JSON array or {{'events': [...]}} in {path}")


def fetch_events_from_supabase(household_id: str, time_window_days: int = 7) -> list[dict]:
    """Fetch real events from Supabase for household. Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."""
    from datetime import datetime, timedelta, timezone
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set to fetch real events")
    from supabase import create_client
    client = create_client(url, key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=time_window_days)
    q = client.table("events").select("*, sessions!inner(household_id)").eq("sessions.household_id", household_id)
    q = q.gte("ts", start.isoformat()).lte("ts", end.isoformat())
    r = q.order("ts").execute()
    rows = list(r.data or [])
    # Normalize to pipeline shape: session_id, device_id, ts, seq, event_type, payload
    out = []
    for ev in rows:
        out.append({
            "session_id": ev.get("session_id"),
            "device_id": ev.get("device_id"),
            "ts": ev.get("ts"),
            "seq": ev.get("seq", 0),
            "event_type": ev.get("event_type", ""),
            "payload": ev.get("payload") or {},
        })
    return out


def run_replay(scenario_events: list[dict], household_id: str = "replay-hh", threshold: float = 0.5) -> dict:
    """Run pipeline on event prefixes until risk exceeds threshold; return time_to_flag and metrics."""
    from api.pipeline import run_pipeline

    events = list(scenario_events)
    if not events:
        return {"time_to_flag_sec": None, "flagged": False, "num_events": 0, "message": "no events"}

    first_ts = _ts_to_float(events[0])
    for i in range(1, len(events) + 1):
        prefix = events[:i]
        result = run_pipeline(household_id, prefix, None, None)
        risk_scores = result.get("risk_scores") or []
        above = [r for r in risk_scores if (r.get("score") or 0) >= threshold]
        if above:
            flag_ts = _ts_to_float(events[i - 1])
            time_to_flag_sec = flag_ts - first_ts
            return {
                "time_to_flag_sec": time_to_flag_sec,
                "flagged": True,
                "num_events_to_flag": i,
                "num_events": len(events),
                "max_score": max(r.get("score", 0) for r in risk_scores),
                "threshold": threshold,
            }
    return {
        "time_to_flag_sec": None,
        "flagged": False,
        "num_events": len(events),
        "message": "threshold never exceeded",
        "threshold": threshold,
    }


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        description="Replay events (real or synthetic) and compute time_to_flag",
        epilog="Real: --events-file or --household-id. Synthetic: --scenario (used only when no real source).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write metrics JSON here")
    parser.add_argument("--events-file", type=Path, default=None, help="Load real events from JSON file ([{...}] or {\"events\": [...]})")
    parser.add_argument("--household-id", type=str, default=None, help="Fetch real events from Supabase for this household (requires SUPABASE_* env)")
    parser.add_argument("--time-window-days", type=int, default=7, help="Days of events to fetch when using --household-id")
    parser.add_argument("--scenario", type=str, default="scam", choices=["scam", "baseline"], help="Synthetic scenario only when no --events-file or --household-id")
    parser.add_argument("--threshold", type=float, default=0.5, help="Risk score threshold for flag")
    args = parser.parse_args()

    source_label = "synthetic"
    household_id = "replay-hh"

    if args.events_file is not None:
        if not args.events_file.is_file():
            print(f"Error: events file not found: {args.events_file}", file=sys.stderr)
            sys.exit(1)
        events, source_label, hh_from_file = load_events_from_file(args.events_file)
        if hh_from_file:
            household_id = hh_from_file
    elif args.household_id:
        try:
            events = fetch_events_from_supabase(args.household_id, args.time_window_days)
            source_label = f"supabase:{args.household_id} (last {args.time_window_days}d)"
            household_id = args.household_id
        except Exception as e:
            print(f"Error fetching real events: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        from scripts.synthetic_scenarios import scam_scenario_events, baseline_normal_events
        if args.scenario == "scam":
            events = scam_scenario_events("replay-session", "replay-device", day_offset=0)
        else:
            events = baseline_normal_events("replay-session", "replay-device", day_offset=0)

    if not events:
        print("No events to replay.", file=sys.stderr)
        sys.exit(1)

    # Replay in chronological order
    events = sorted(events, key=_ts_to_float)

    metrics = run_replay(events, household_id=household_id, threshold=args.threshold)
    metrics["source"] = source_label

    print(json.dumps(metrics, indent=2))
    if metrics.get("time_to_flag_sec") is not None:
        print(f"time_to_flag: {metrics['time_to_flag_sec']:.2f} sec", file=sys.stderr)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
