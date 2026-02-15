#!/usr/bin/env python3
"""
Build demo-mode fixtures from the elderly conversation dataset so the frontend
shows 35,000 data points when demo mode is on (bottom-left toggle).

Reads data/elderly_conversations/sessions.json and summaries.json, then writes:
  apps/web/public/fixtures/sessions.json   (first page of sessions, total: 35000)
  apps/web/public/fixtures/summaries.json  (weekly summaries reflecting 35k conversations)
  apps/web/public/fixtures/risk_signals.json (sample risk signals, total ~120)

Run from repo root after generating elderly data:
  python3 scripts/build_demo_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "elderly_conversations"
OUT_DIR = REPO_ROOT / "apps" / "web" / "public" / "fixtures"
TOTAL_SESSIONS = 35_000


def main() -> None:
    if not DATA_DIR.exists():
        print("Run first: python3 scripts/generate_elderly_conversations.py", file=__import__("sys").stderr)
        raise SystemExit(1)

    with open(DATA_DIR / "sessions.json") as f:
        data = json.load(f)
    sessions = data["sessions"]
    device_ids = data.get("device_ids", [])

    with open(DATA_DIR / "summaries.json") as f:
        sum_data = json.load(f)
    summaries_list = sum_data.get("summaries", [])
    session_to_summary = {s["session_id"]: s.get("summary_text", "") for s in summaries_list}

    # First page of sessions (25) with summary_text; total 35000
    page_size = 25
    session_items = []
    for s in sessions[:page_size]:
        session_items.append({
            "id": s["id"],
            "device_id": s["device_id"],
            "started_at": s["started_at"],
            "ended_at": s["ended_at"] or s["started_at"],
            "mode": s.get("mode", "offline"),
            "consent_state": s.get("consent_state", {}),
            "summary_text": session_to_summary.get(s["id"]),
        })
    sessions_fixture = {"sessions": session_items, "total": TOTAL_SESSIONS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "sessions.json", "w") as f:
        json.dump(sessions_fixture, f, indent=2)
    print("Wrote", OUT_DIR / "sessions.json", "total:", TOTAL_SESSIONS)

    # Weekly summaries (period-scoped) that reflect 35k conversations
    weekly = [
        {"id": str(uuid4()), "period_start": "2024-02-14T00:00:00.000Z", "period_end": "2024-02-20T23:59:59.000Z", "summary_text": "This week: ~670 conversations. Routine reminders, family calls, and a few Medicare-related questions.", "summary_json": {"signal_count": 4}},
        {"id": str(uuid4()), "period_start": "2024-06-01T00:00:00.000Z", "period_end": "2024-06-07T23:59:59.000Z", "summary_text": "~672 conversations. One potential scam pattern (gift card request) flagged; caregiver notified.", "summary_json": {"signal_count": 1}},
        {"id": str(uuid4()), "period_start": "2024-10-01T00:00:00.000Z", "period_end": "2024-10-07T23:59:59.000Z", "summary_text": "~670 conversations. Mix of routine check-ins and doctor/pharmacy reminders. Two risk signals (IRS-themed call, account verification).", "summary_json": {"signal_count": 2}},
        {"id": str(uuid4()), "period_start": "2025-01-01T00:00:00.000Z", "period_end": "2025-01-07T23:59:59.000Z", "summary_text": "First week of year: ~672 conversations. Social Security and Medicare mentions; one urgency-language signal open.", "summary_json": {"signal_count": 1}},
        {"id": str(uuid4()), "period_start": "2025-02-08T00:00:00.000Z", "period_end": "2025-02-14T23:59:59.000Z", "summary_text": "Past year total: 35,000+ conversations across all devices. Routine daily use with periodic risk review.", "summary_json": {"signal_count": 0}},
    ]
    with open(OUT_DIR / "summaries.json", "w") as f:
        json.dump(weekly, f, indent=2)
    print("Wrote", OUT_DIR / "summaries.json")

    # Risk signals: sample of ~30, total ~120 (realistic for 35k conversations)
    risk_types = ["relational_anomaly", "urgency_language", "sensitive_request", "unusual_payee", "motif_medicare", "motif_irs"]
    statuses = ["open", "acknowledged", "dismissed"]
    signals = []
    for i in range(32):
        signals.append({
            "id": str(uuid4()),
            "ts": f"2025-0{(i % 9) + 1}-{(i % 28) + 1}T{(10 + i % 12):02d}:00:00.000Z",
            "signal_type": risk_types[i % len(risk_types)],
            "severity": (i % 5) + 1,
            "score": round(0.35 + (i % 60) / 100, 2),
            "status": statuses[i % 3],
            "summary": "Medicare/IRS-themed call; possible sensitive request." if i % 4 == 0 else "Routine contact; low concern.",
        })
    risk_fixture = {"signals": signals, "total": 127}
    with open(OUT_DIR / "risk_signals.json", "w") as f:
        json.dump(risk_fixture, f, indent=2)
    print("Wrote", OUT_DIR / "risk_signals.json", "total: 127")


if __name__ == "__main__":
    main()
