"""Tests for centralized time_utils (datetime, ISO, unix, tz)."""
from __future__ import annotations

from datetime import datetime, timezone


def test_time_utils_accepts_datetime_iso_unix_and_handles_tz() -> None:
    from domain.utils.time_utils import ts_to_float

    # None -> 0.0
    assert ts_to_float(None) == 0.0
    # Unix int/float
    assert ts_to_float(1609459200) == 1609459200.0
    assert ts_to_float(1609459200.5) == 1609459200.5
    # ISO string with Z
    t = ts_to_float("2021-01-01T00:00:00Z")
    assert t == 1609459200.0 or abs(t - 1609459200.0) < 1
    # ISO with +00:00
    t2 = ts_to_float("2021-01-01T00:00:00+00:00")
    assert abs(t2 - 1609459200.0) < 1
    # datetime
    dt = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert abs(ts_to_float(dt) - 1609459200.0) < 1
