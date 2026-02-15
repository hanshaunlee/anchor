#!/usr/bin/env python3
"""
Stress / smoke test for Automation Center flows.
Calls investigation dry_run, agents/status, agents/trace, outreach/candidates, preview, (optional) send.
Outputs JSON report: endpoints succeeded, expected fields present, counts.
Requires: API running (e.g. uvicorn), valid auth token for caregiver/admin.
Usage:
  export ANCHOR_TOKEN=...  # or use demo
  python scripts/stress_automation_center.py [--base URL] [--no-send]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)

DEFAULT_BASE = os.environ.get("ANCHOR_API_BASE", "http://localhost:8000")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stress Automation Center endpoints")
    ap.add_argument("--base", default=DEFAULT_BASE, help="API base URL")
    ap.add_argument("--no-send", action="store_true", help="Do not call outreach/send")
    ap.add_argument("--token", default=os.environ.get("ANCHOR_TOKEN"), help="Bearer token")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    report: dict[str, Any] = {
        "endpoints": {},
        "expected_fields": {},
        "counts": {},
        "errors": [],
    }

    # 1) POST /investigation/run dry_run
    try:
        r = requests.post(
            f"{base}/investigation/run",
            json={"dry_run": True, "use_demo_events": True},
            headers=headers,
            timeout=60,
        )
        report["endpoints"]["investigation_run_dry"] = r.status_code == 200
        if r.status_code == 200:
            d = r.json()
            report["expected_fields"]["investigation_run"] = all(
                k in d for k in ("ok", "supervisor_run_id", "created_signal_ids", "summary_json", "step_trace", "outreach_candidates")
            )
            report["counts"]["created_signal_ids"] = len(d.get("created_signal_ids") or [])
            report["counts"]["outreach_candidates"] = len(d.get("outreach_candidates") or [])
            report["counts"]["step_trace_len"] = len(d.get("step_trace") or [])
        else:
            report["errors"].append(f"investigation/run: {r.status_code} {r.text[:200]}")
    except Exception as e:
        report["endpoints"]["investigation_run_dry"] = False
        report["errors"].append(f"investigation/run: {e!r}")

    # 2) GET /agents/status
    try:
        r = requests.get(f"{base}/agents/status", headers=headers, timeout=10)
        report["endpoints"]["agents_status"] = r.status_code == 200
        if r.status_code == 200:
            d = r.json()
            report["expected_fields"]["agents_status"] = "agents" in d and isinstance(d["agents"], list)
            if d.get("agents"):
                first = d["agents"][0]
                report["expected_fields"]["agents_status_last_run_id"] = "last_run_id" in first
        else:
            report["errors"].append(f"agents/status: {r.status_code}")
    except Exception as e:
        report["endpoints"]["agents_status"] = False
        report["errors"].append(f"agents/status: {e!r}")

    # 3) GET /agents/trace (if we have a run_id from step 1 we could use it; optional)
    run_id = None
    if report.get("endpoints", {}).get("investigation_run_dry"):
        try:
            r = requests.post(
                f"{base}/investigation/run",
                json={"dry_run": True, "use_demo_events": True},
                headers=headers,
                timeout=60,
            )
            if r.status_code == 200:
                run_id = r.json().get("supervisor_run_id")
        except Exception:
            pass
    if run_id:
        try:
            r = requests.get(
                f"{base}/agents/trace",
                params={"run_id": run_id, "agent_name": "supervisor"},
                headers=headers,
                timeout=10,
            )
            report["endpoints"]["agents_trace"] = r.status_code == 200
            if r.status_code == 200:
                d = r.json()
                report["expected_fields"]["agents_trace"] = "step_trace" in d or "id" in d
        except Exception as e:
            report["endpoints"]["agents_trace"] = False
            report["errors"].append(f"agents/trace: {e!r}")
    else:
        report["endpoints"]["agents_trace"] = None
        report["expected_fields"]["agents_trace"] = None

    # 4) GET /actions/outreach/candidates
    try:
        r = requests.get(f"{base}/actions/outreach/candidates", headers=headers, timeout=10)
        report["endpoints"]["outreach_candidates"] = r.status_code == 200
        if r.status_code == 200:
            d = r.json()
            report["expected_fields"]["outreach_candidates"] = "candidates" in d
            report["counts"]["candidates"] = len(d.get("candidates") or [])
        else:
            report["errors"].append(f"outreach/candidates: {r.status_code}")
    except Exception as e:
        report["endpoints"]["outreach_candidates"] = False
        report["errors"].append(f"outreach/candidates: {e!r}")

    # 5) POST /actions/outreach/preview (use first risk_signal from investigation if any)
    risk_signal_id = None
    if report.get("counts", {}).get("created_signal_ids", 0) > 0 and report.get("endpoints", {}).get("investigation_run_dry"):
        try:
            r2 = requests.post(
                f"{base}/investigation/run",
                json={"dry_run": True, "use_demo_events": True},
                headers=headers,
                timeout=60,
            )
            if r2.status_code == 200:
                ids = r2.json().get("created_signal_ids") or []
                if ids:
                    risk_signal_id = ids[0]
        except Exception:
            pass
    if not risk_signal_id and report.get("counts", {}).get("candidates", 0) > 0:
        try:
            r2 = requests.get(f"{base}/actions/outreach/candidates", headers=headers, timeout=10)
            if r2.status_code == 200:
                cands = r2.json().get("candidates") or []
                if cands:
                    risk_signal_id = cands[0].get("risk_signal_id")
        except Exception:
            pass
    if risk_signal_id:
        try:
            r = requests.post(
                f"{base}/actions/outreach/preview",
                json={"risk_signal_id": risk_signal_id},
                headers=headers,
                timeout=30,
            )
            report["endpoints"]["outreach_preview"] = r.status_code == 200
            if r.status_code == 200:
                d = r.json()
                report["expected_fields"]["outreach_preview"] = "preview" in d
        except Exception as e:
            report["endpoints"]["outreach_preview"] = False
            report["errors"].append(f"outreach/preview: {e!r}")
    else:
        report["endpoints"]["outreach_preview"] = None

    # 6) POST /actions/outreach/send (optional, skip by default)
    if not args.no_send and risk_signal_id:
        try:
            r = requests.post(
                f"{base}/actions/outreach/send",
                json={"risk_signal_id": risk_signal_id},
                headers=headers,
                timeout=30,
            )
            report["endpoints"]["outreach_send"] = r.status_code == 200
            if r.status_code == 200:
                d = r.json()
                report["expected_fields"]["outreach_send"] = "sent" in d or "suppressed" in d
        except Exception as e:
            report["endpoints"]["outreach_send"] = False
            report["errors"].append(f"outreach/send: {e!r}")
    else:
        report["endpoints"]["outreach_send"] = None

    print(json.dumps(report, indent=2))
    succeeded = sum(1 for v in report["endpoints"].values() if v is True)
    total = sum(1 for v in report["endpoints"].values() if v is not None)
    return 0 if (total > 0 and succeeded == total) else 1


if __name__ == "__main__":
    sys.exit(main())
