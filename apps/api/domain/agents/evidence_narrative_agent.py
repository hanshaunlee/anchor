"""
Evidence Narrative Agent: evidence-grounded narrative with redaction.
Deterministic template (default) + optional LLM; updates risk_signals.explanation.summary and narrative.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from domain.agents.base import AgentContext, persist_agent_run, step

logger = logging.getLogger(__name__)

SIGNAL_LIMIT = 20
EVIDENCE_ONLY_BADGE = "Evidence-only"


def _consent_state(supabase: Any, household_id: str) -> dict[str, Any]:
    try:
        r = (
            supabase.table("sessions")
            .select("consent_state")
            .eq("household_id", household_id)
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        )
        if r.data and len(r.data) > 0:
            return r.data[0].get("consent_state") or {}
    except Exception:
        pass
    return {}


def _redact_timeline_snippet(snippet: list[dict], forbid_text: bool) -> list[dict]:
    if not forbid_text:
        return snippet
    return [
        {**item, "text_preview": "(redacted)"} if isinstance(item, dict) else item
        for item in snippet
    ]


def _redact_entity_display(entity_id: str, entity_type: str, canonical: str, forbid_canonical: bool) -> str:
    if not forbid_canonical or not canonical:
        return canonical or f"{entity_type}:{entity_id[:8]}"
    h = hashlib.sha256(canonical.encode()).hexdigest()[-6:]
    return f"{entity_type}:...{h}"


def _build_evidence_bundle(
    signal_id: str,
    expl: dict,
    supabase: Any,
    household_id: str,
    consent_share_text: bool,
    consent_share_entity_canonical: bool,
) -> dict[str, Any]:
    """Build JSON evidence bundle for narrative: subgraph, motifs, timeline, entities (redacted)."""
    subgraph = expl.get("model_subgraph") or expl.get("subgraph") or {}
    nodes = list(subgraph.get("nodes") or [])
    edges = list(subgraph.get("edges") or [])
    motif_tags = list(expl.get("motif_tags") or [])
    timeline_snippet = _redact_timeline_snippet(
        list(expl.get("timeline_snippet") or []),
        forbid_text=not consent_share_text,
    )
    entity_ids = [n.get("id") for n in nodes if n.get("id")]
    entity_displays: dict[str, str] = {}
    if entity_ids and supabase:
        try:
            r = supabase.table("entities").select("id, entity_type, canonical").eq("household_id", household_id).in_("id", entity_ids).execute()
            for e in (r.data or []):
                eid = str(e.get("id", ""))
                entity_displays[eid] = _redact_entity_display(
                    eid,
                    e.get("entity_type", "entity"),
                    e.get("canonical", ""),
                    forbid_canonical=not consent_share_entity_canonical,
                )
        except Exception:
            pass
    return {
        "signal_id": signal_id,
        "entity_ids": entity_ids,
        "entity_displays": entity_displays,
        "nodes": nodes,
        "edges": edges,
        "motif_tags": motif_tags,
        "timeline_snippet": timeline_snippet,
        "redacted": not consent_share_text or not consent_share_entity_canonical,
    }


def _deterministic_narrative(bundle: dict[str, Any]) -> str:
    """Evidence-only narrative from template; no hallucination."""
    parts = []
    parts.append("What happened: Timeline of " + str(len(bundle.get("timeline_snippet") or [])) + " event(s) in evidence.")
    motifs = bundle.get("motif_tags") or []
    if motifs:
        parts.append("Why suspicious: Patterns observed — " + "; ".join(motifs[:5]) + ".")
    nodes = bundle.get("nodes") or []
    edges = bundle.get("edges") or []
    parts.append("Who/what involved: " + str(len(nodes)) + " entity(ies), " + str(len(edges)) + " link(s) in evidence subgraph.")
    displays = bundle.get("entity_displays") or {}
    if displays:
        parts.append("Entities (evidence-only): " + ", ".join(displays.values())[:200] + ("…" if len(str(displays)) > 200 else ""))
    parts.append("Recommended next steps: Review recommended_action checklist for this alert.")
    parts.append("Confidence: Based solely on evidence bundle; no external facts added.")
    return " ".join(parts)


def _llm_narrative_if_available(bundle: dict[str, Any]) -> str | None:
    """Optional LLM narrative; must reference only entity ids in bundle. If invalid, return None."""
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = (
            "You are a caregiver-facing assistant. Generate a short narrative (2–4 sentences) based ONLY on the following JSON evidence. "
            "DO NOT add any fact not present in the evidence. Only reference entity IDs that appear in the bundle. "
            "Include: what happened (timeline), why it is suspicious (motifs), who/what is involved (entities from entity_displays), and recommended next steps.\n\n"
            + json.dumps({k: v for k, v in bundle.items() if k != "signal_id"}, default=str)
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return None
        allowed_ids = set(bundle.get("entity_ids") or [])
        for n in bundle.get("nodes") or []:
            allowed_ids.add(str(n.get("id", "")))
        if allowed_ids and any(
            rid not in text and rid not in "".join(str(v) for v in (bundle.get("entity_displays") or {}).values())
            for rid in allowed_ids
        ):
            return None
        return text
    except Exception as e:
        logger.debug("LLM narrative skipped: %s", e)
        return None


def run_evidence_narrative_agent(
    household_id: str,
    supabase: Any | None = None,
    dry_run: bool = False,
    *,
    risk_signal_ids: list[str] | None = None,
    risk_signal_id: str | None = None,
) -> dict[str, Any]:
    """
    Generate evidence-grounded narratives for open signals (or given ids). Limit 20.
    Applies consent redaction; deterministic template by default; optional LLM with validation.
    Persists agent_run and updates risk_signals.explanation.summary and narrative.
    """
    step_trace: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()
    ctx = AgentContext(household_id, supabase, dry_run=dry_run)

    if not supabase:
        step_trace.append({
            "step": "fetch_signals",
            "status": "skip",
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "notes": "no_supabase",
        })
        summary = {"updated": 0, "signals_processed": 0, "reason": "no_supabase"}
        ended_at = datetime.now(timezone.utc).isoformat()
        return {
            "step_trace": step_trace,
            "summary_json": summary,
            "status": "ok",
            "started_at": started_at,
            "ended_at": ended_at,
            "run_id": None,
        }

    consent_state = _consent_state(supabase, household_id)
    consent_share_text = consent_state.get("share_with_caregiver", True)
    consent_share_entity_canonical = consent_state.get("share_with_caregiver", True)

    with step(ctx, step_trace, "fetch_signals"):
        ids_filter = risk_signal_ids or ([risk_signal_id] if risk_signal_id else None)
        q = (
            supabase.table("risk_signals")
            .select("id, explanation")
            .eq("household_id", household_id)
            .eq("status", "open")
        )
        if ids_filter:
            q = q.in_("id", ids_filter[:SIGNAL_LIMIT])
        else:
            q = q.order("ts", desc=True).limit(SIGNAL_LIMIT)
        r = q.execute()
        signals = r.data or []
        step_trace[-1]["outputs_count"] = len(signals)
        step_trace[-1]["notes"] = f"{len(signals)} open signals"

    per_signal_results: list[dict] = []
    updated = 0

    with step(ctx, step_trace, "gather_evidence_and_narrative"):
        for s in signals:
            expl = s.get("explanation") or {}
            subgraph = expl.get("model_subgraph") or expl.get("subgraph") or {}
            motifs = expl.get("motif_tags") or []
            nodes = subgraph.get("nodes") or []
            edges = subgraph.get("edges") or []
            if not nodes and not edges and not motifs:
                per_signal_results.append({"signal_id": s["id"], "status": "skipped", "reason": "no_evidence"})
                continue
            bundle = _build_evidence_bundle(
                s["id"], expl, supabase, household_id,
                consent_share_text, consent_share_entity_canonical,
            )
            narrative_llm = _llm_narrative_if_available(bundle)
            if narrative_llm:
                narrative = narrative_llm
            else:
                narrative = _deterministic_narrative(bundle)
            summary_text = expl.get("summary") or narrative[:300]
            new_expl = {
                **expl,
                "summary": summary_text,
                "narrative": narrative,
                "narrative_evidence_only": True,
                "narrative_agent_run": datetime.now(timezone.utc).isoformat(),
            }
            if not dry_run:
                try:
                    supabase.table("risk_signals").update({"explanation": new_expl}).eq("id", s["id"]).eq("household_id", household_id).execute()
                    updated += 1
                    per_signal_results.append({"signal_id": s["id"], "status": "updated", "narrative_length": len(narrative)})
                except Exception as e:
                    step_trace[-1]["error"] = str(e)
                    per_signal_results.append({"signal_id": s["id"], "status": "error", "error": str(e)})
            else:
                updated += 1
                per_signal_results.append({"signal_id": s["id"], "status": "dry_run", "narrative_length": len(narrative)})
        step_trace[-1]["outputs_count"] = updated
        step_trace[-1]["notes"] = f"updated {updated} of {len(signals)}"

    summary_json = {
        "updated": updated,
        "signals_processed": len(signals),
        "per_signal": per_signal_results,
        "artifact_refs": {"risk_signal_ids": [s["id"] for s in signals]},
    }
    ended_at = datetime.now(timezone.utc).isoformat()
    run_id = persist_agent_run(
        supabase, household_id, "evidence_narrative",
        started_at=started_at, ended_at=ended_at, status="completed",
        step_trace=step_trace, summary_json=summary_json, dry_run=dry_run,
    )
    return {
        "step_trace": step_trace,
        "summary_json": summary_json,
        "status": "ok",
        "started_at": started_at,
        "ended_at": ended_at,
        "run_id": run_id,
    }
