"""Tests: protection overview shape, watchlist fingerprint/batch dedupe, ring fingerprint."""
from __future__ import annotations

import pytest


def test_ring_fingerprint_deterministic() -> None:
    from domain.rings.fingerprint import ring_fingerprint

    ids = ["e1", "e2", "e3"]
    fp1 = ring_fingerprint(ids)
    fp2 = ring_fingerprint(reversed(ids))
    assert fp1 == fp2
    assert len(fp1) == 64  # sha256 hex


def test_ring_fingerprint_empty() -> None:
    from domain.rings.fingerprint import ring_fingerprint

    assert len(ring_fingerprint([])) == 64
    assert ring_fingerprint([]) == ring_fingerprint([])


def test_jaccard_overlap() -> None:
    from domain.rings.fingerprint import jaccard_overlap

    assert jaccard_overlap({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard_overlap({"a", "b"}, {"a", "b", "c"}) == 2 / 3
    assert jaccard_overlap(set(), set()) == 0.0
    assert jaccard_overlap({"a"}, {"b"}) == 0.0


def test_watchlist_fingerprint_deterministic() -> None:
    from domain.watchlists.normalize import watchlist_fingerprint

    fp = watchlist_fingerprint("topic", "risky_topic", "risky_topic", "medicare")
    assert len(fp) == 64
    assert fp == watchlist_fingerprint("topic", "risky_topic", "risky_topic", "medicare")


def test_protection_overview_shape() -> None:
    """Protection overview response has watchlist_summary, rings_summary, reports_summary."""
    from api.routers.protection import ProtectionOverview, WatchlistSummary, WatchlistItemDisplay, RingSummary, ReportSummary

    overview = ProtectionOverview(
        watchlist_summary=WatchlistSummary(total=0, by_category={}, items=[]),
        rings_summary=[],
        reports_summary=[],
        last_updated_at=None,
        data_freshness={},
    )
    assert overview.watchlist_summary.total == 0
    assert isinstance(overview.rings_summary, list)
    assert isinstance(overview.reports_summary, list)

    with_items = ProtectionOverview(
        watchlist_summary=WatchlistSummary(
            total=2,
            by_category={"topic": 1, "contact": 1},
            items=[
                WatchlistItemDisplay(id="1", category="topic", type="risky_topic", display_label="Topic", display_value="medicare", explanation="x", priority=2, score=0.8, source_agent="financial", evidence_signal_ids=[]),
                WatchlistItemDisplay(id="2", category="contact", type="new_contact", display_label="Contact", display_value="+1 (555) 123-4567", explanation="y", priority=3, score=None, source_agent="supervisor", evidence_signal_ids=[]),
            ],
        ),
        rings_summary=[
            RingSummary(id="r1", household_id="hh1", created_at="2025-01-01T00:00:00Z", updated_at="2025-01-01T00:00:00Z", score=0.85, summary_label="Cluster", summary_text="Desc", members_count=4, meta={}),
        ],
        reports_summary=[ReportSummary(kind="Narrative", last_run_at="2025-01-01", last_run_id="run1", summary="Ok", status="ok")],
        last_updated_at="2025-01-01T00:00:00Z",
        data_freshness={},
    )
    assert with_items.watchlist_summary.total == 2
    assert len(with_items.rings_summary) == 1
    assert with_items.rings_summary[0].summary_label == "Cluster"


def test_upsert_watchlist_batch_dedupe_supersede() -> None:
    """With mock supabase, upsert_watchlist_batch normalizes and uses fingerprint; no DB so we only test the mapping."""
    from domain.watchlists.service import _map_legacy_to_item

    row = _map_legacy_to_item(
        "hh1",
        "batch1",
        {"watch_type": "entity_pattern", "pattern": {"keywords": ["medicare", "irs"]}, "reason": "Risky topics", "priority": 2},
        "financial_security",
        "run1",
        None,
    )
    assert row["category"] == "topic"
    assert row["batch_id"] == "batch1"
    assert row["fingerprint"]
    assert "medicare" in (row.get("display_value") or "").lower() or "irs" in (row.get("display_value") or "").lower()
