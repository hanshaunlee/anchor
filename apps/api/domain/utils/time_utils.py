"""
Centralized timestamp handling for pipeline, financial agent, motifs, graph builder, scripts.
Accepts: naive datetime, tz-aware datetime, ISO string, unix int, float.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def ts_to_float(ts: datetime | str | int | float | None) -> float:
    """
    Normalize any timestamp representation to Unix float (seconds since epoch).
    - None -> 0.0
    - int/float -> float (assumed Unix)
    - str -> parsed as ISO (Z -> +00:00), then timestamp
    - datetime (naive or aware) -> timestamp (naive treated as UTC)
    """
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            return t.timestamp()
        except (ValueError, TypeError):
            return 0.0
    if hasattr(ts, "timestamp"):
        t = ts
        if getattr(t, "tzinfo", None) is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.timestamp()
    return 0.0


def event_ts_to_float(event: dict[str, Any]) -> float:
    """Extract timestamp from an event dict (key 'ts') and return as float."""
    return ts_to_float(event.get("ts") if isinstance(event, dict) else None)


def float_to_datetime(f: float) -> datetime:
    """Convert Unix float to timezone-aware UTC datetime."""
    return datetime.fromtimestamp(float(f), tz=timezone.utc)
