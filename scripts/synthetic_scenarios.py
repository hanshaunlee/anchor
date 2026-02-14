#!/usr/bin/env python3
"""
Synthetic scenario generator: baseline normal routine + scam scenario.
Produces realistic event streams (session_id, device_id, ts, seq, event_type, payload).
Output: JSON file or stdout for ingestion via POST /ingest/events.
"""
import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from uuid import uuid4

def make_ts(day_offset: int, hour: int, minute: int, second_offset: int = 0) -> str:
    """Return ISO timestamp for (day_offset, hour, minute) + second_offset seconds."""
    t = datetime.now(timezone.utc) - timedelta(days=day_offset)
    t = t.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if second_offset:
        t = t + timedelta(seconds=second_offset)
    return t.isoformat()


def baseline_normal_events(session_id: str, device_id: str, day_offset: int = 0) -> list[dict]:
    """Routine: wake, final_asr (reminder), intent (reminder), no sensitive entities."""
    events = [
        {"session_id": session_id, "device_id": device_id, "ts": make_ts(day_offset, 9, 0, 0), "seq": 0, "event_type": "wake", "payload_version": 1, "payload": {}},
        {"session_id": session_id, "device_id": device_id, "ts": make_ts(day_offset, 9, 0, 10), "seq": 1, "event_type": "final_asr", "payload_version": 1, "payload": {"text": "Remind me to take medicine at 5", "confidence": 0.9, "speaker": {"role": "elder"}}},
        {"session_id": session_id, "device_id": device_id, "ts": make_ts(day_offset, 9, 0, 20), "seq": 2, "event_type": "intent", "payload_version": 1, "payload": {"name": "reminder", "slots": {"time": "5pm"}, "confidence": 0.85}},
    ]
    return events


def scam_scenario_events(session_id: str, device_id: str, day_offset: int = 0) -> list[dict]:
    """Scam: unknown phone, urgency (Medicare), repeated attempts, request for sensitive info.
    Each event has a distinct timestamp (base + 15s per seq) so time_to_flag is meaningful in replay."""
    events = [
        {"session_id": session_id, "device_id": device_id, "ts": make_ts(day_offset, 14, 30, 0), "seq": 0, "event_type": "wake", "payload_version": 1, "payload": {}},
        {"session_id": session_id, "device_id": device_id, "ts": make_ts(day_offset, 14, 30, 15), "seq": 1, "event_type": "final_asr", "payload_version": 1, "payload": {"text": "Someone called about Medicare", "confidence": 0.88, "speaker": {"role": "elder"}}},
        {"session_id": session_id, "device_id": device_id, "ts": make_ts(day_offset, 14, 30, 30), "seq": 2, "event_type": "intent", "payload_version": 1, "payload": {"name": "call_log", "slots": {"topic": "Medicare", "phone_number": "+15551234567"}, "confidence": 0.8}},
        {"session_id": session_id, "device_id": device_id, "ts": make_ts(day_offset, 14, 30, 45), "seq": 3, "event_type": "final_asr", "payload_version": 1, "payload": {"text": "They said I need to give my Social Security number", "confidence": 0.85, "speaker": {"role": "elder"}}},
        {"session_id": session_id, "device_id": device_id, "ts": make_ts(day_offset, 14, 30, 60), "seq": 4, "event_type": "intent", "payload_version": 1, "payload": {"name": "sensitive_request", "slots": {"topic": "Medicare", "info_type": "ssn"}, "confidence": 0.9}},
    ]
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--household-id", type=str, default=None, help="Household UUID (for reference)")
    parser.add_argument("--output", type=str, default=None, help="Write events to JSON file")
    parser.add_argument("--sessions", type=int, default=3, help="Number of sessions (mix of normal + scam)")
    args = parser.parse_args()
    household_id = args.household_id or str(uuid4())
    device_id = str(uuid4())
    all_events = []
    for i in range(args.sessions):
        session_id = str(uuid4())
        if i % 2 == 0:
            all_events.extend(baseline_normal_events(session_id, device_id, day_offset=i))
        else:
            all_events.extend(scam_scenario_events(session_id, device_id, day_offset=i))
    out = {"household_id": household_id, "device_id": device_id, "events": all_events}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {len(all_events)} events to {args.output}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
