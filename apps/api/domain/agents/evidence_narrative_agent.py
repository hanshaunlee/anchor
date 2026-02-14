"""Evidence Narrative Agent: model_subgraph + motifs -> caregiver-readable summary; store in risk_signals.explanation.summary."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def run_evidence_narrative_agent(
    household_id: str,
    risk_signal_id: str | None = None,
    supabase: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    For risk signals with model_subgraph + motifs, produce a short caregiver-readable narrative
    grounded only in evidence pointers; store in risk_signals.explanation.summary.
    If risk_signal_id given, process that one; else process recent open signals for household.
    """
    step_trace: list[dict] = []
    started = datetime.now(timezone.utc).isoformat()

    step_trace.append({"step": "fetch_signals", "status": "ok", "ts": started})
    updated = 0

    if not supabase:
        step_trace.append({"step": "narrative", "status": "skip", "reason": "no_supabase"})
        return {
            "step_trace": step_trace,
            "summary_json": {"updated": 0},
            "status": "ok",
            "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }

    try:
        q = (
            supabase.table("risk_signals")
            .select("id, explanation")
            .eq("household_id", household_id)
            .eq("status", "open")
        )
        if risk_signal_id:
            q = q.eq("id", risk_signal_id)
        else:
            q = q.order("ts", desc=True).limit(20)
        r = q.execute()
        signals = r.data or []
    except Exception as e:
        logger.warning("Evidence narrative fetch failed: %s", e)
        step_trace.append({"step": "fetch_signals", "status": "error", "error": str(e)})
        return {
            "step_trace": step_trace,
            "summary_json": {"error": str(e)},
            "status": "error",
            "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }

    for s in signals:
        expl = s.get("explanation") or {}
        subgraph = expl.get("model_subgraph") or expl.get("subgraph") or {}
        motifs = expl.get("motif_tags") or []
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        if not nodes and not motifs:
            continue
        # Build narrative: "N entities and M links suggest risk. Patterns: A; B; C."
        parts = []
        if nodes:
            parts.append(f"{len(nodes)} entity(ies) and {len(edges)} link(s) in evidence subgraph.")
        if motifs:
            parts.append("Patterns: " + "; ".join(motifs[:5]) + ".")
        summary = " ".join(parts) if parts else expl.get("summary", "No narrative generated.")
        if not dry_run:
            try:
                new_expl = {**expl, "summary": summary, "narrative_agent_run": datetime.now(timezone.utc).isoformat()}
                supabase.table("risk_signals").update({"explanation": new_expl}).eq("id", s["id"]).execute()
                updated += 1
            except Exception as e:
                step_trace.append({"step": "update", "signal_id": s["id"], "status": "error", "error": str(e)})
        else:
            updated += 1

    step_trace.append({"step": "narrative", "status": "ok", "signals_processed": len(signals), "updated": updated})
    return {
        "step_trace": step_trace,
        "summary_json": {"updated": updated, "signals_processed": len(signals)},
        "status": "ok",
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
