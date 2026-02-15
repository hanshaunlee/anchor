"""Risk signal compound upsert: same risk (by fingerprint) compounds score and refreshes; old risks phase out by updated_at."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

from typing import Any

logger = logging.getLogger(__name__)

# Compounding: new_score = min(1.0, old * decay + incoming * weight). Repeated hits lift score; cap at 1.
COMPOUND_DECAY = 0.85  # retain 85% of previous score
COMPOUND_WEIGHT = 0.25  # add 25% of incoming score


def risk_signal_fingerprint(signal_type: str, explanation: dict[str, Any]) -> str:
    """Stable hash so the same risk (same type + same entity/motif context) gets the same fingerprint."""
    motifs = explanation.get("motif_tags") or explanation.get("semantic_pattern_tags") or []
    key_parts = [
        signal_type,
        "|",
        ",".join(sorted(motifs)) if isinstance(motifs, list) else str(motifs),
        "|",
        str(explanation.get("summary", "")),
    ]
    # Include entity index or first node id so same motif on different entity = different risk
    subgraph = explanation.get("subgraph") or explanation.get("model_subgraph") or {}
    nodes = subgraph.get("nodes") or []
    if nodes and isinstance(nodes[0], dict):
        key_parts.append(nodes[0].get("id", ""))
    else:
        key_parts.append(explanation.get("independence", {}).get("cluster_id", ""))
    payload = "".join(str(p) for p in key_parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def upsert_risk_signal_compound(
    supabase: Any,
    household_id: str,
    payload: dict[str, Any],
    dry_run: bool = False,
) -> tuple[str | None, bool, float]:
    """
    Upsert by fingerprint: if an open risk with same fingerprint exists, compound score and refresh updated_at.
    Otherwise insert. Returns (risk_signal_id, was_updated, score_used) for broadcast.
    """
    if dry_run:
        return None, False, 0.0
    signal_type = payload.get("signal_type", "risk_signal")
    explanation = payload.get("explanation") or {}
    fingerprint = risk_signal_fingerprint(signal_type, explanation)
    now = datetime.now(timezone.utc).isoformat()
    incoming_score = float(payload.get("score", 0.0))
    incoming_severity = int(payload.get("severity", 2))

    try:
        existing = (
            supabase.table("risk_signals")
            .select("id, score, severity, explanation, updated_at")
            .eq("household_id", household_id)
            .eq("fingerprint", fingerprint)
            .eq("status", "open")
            .limit(1)
            .execute()
        )
        if existing.data and len(existing.data) > 0:
            rec = existing.data[0]
            rid = rec["id"]
            old_score = float(rec.get("score") or 0)
            old_severity = int(rec.get("severity") or 1)
            compounded_score = min(1.0, old_score * COMPOUND_DECAY + incoming_score * COMPOUND_WEIGHT)
            new_severity = max(old_severity, incoming_severity)
            # Merge explanation: keep latest but add hit_count for compounding visibility
            merged_explanation = dict(explanation)
            prev = rec.get("explanation") or {}
            hit_count = (prev.get("compound_hit_count") or 1) + 1
            merged_explanation["compound_hit_count"] = hit_count
            merged_explanation["previous_score"] = round(old_score, 4)
            supabase.table("risk_signals").update({
                "score": round(compounded_score, 4),
                "severity": new_severity,
                "explanation": merged_explanation,
                "recommended_action": payload.get("recommended_action") or {},
                "updated_at": now,
            }).eq("id", rid).execute()
            return str(rid), True, round(compounded_score, 4)
        # Insert new
        row = supabase.table("risk_signals").insert({
            "household_id": household_id,
            "signal_type": signal_type,
            "severity": incoming_severity,
            "score": round(incoming_score, 4),
            "explanation": explanation,
            "recommended_action": payload.get("recommended_action") or {},
            "status": payload.get("status", "open"),
            "fingerprint": fingerprint,
        }).execute()
        if row.data and len(row.data) > 0:
            return row.data[0].get("id"), False, round(incoming_score, 4)
    except Exception as e:
        logger.exception("upsert_risk_signal_compound failed: %s", e)
    return None, False, 0.0
