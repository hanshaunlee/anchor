"""
Evidence Narrative Agent: Investigation Packager.
Eight steps: select signals & fetch evidence, evidence normalization, what-changed diff,
hypothesis generation (optional LLM), caregiver narrative (optional LLM), elder-safe version,
persist into risk_signals + summaries, UI integration.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any

from domain.agents.base import AgentContext, persist_agent_run, persist_agent_run_ctx, step, upsert_summary_ctx

logger = logging.getLogger(__name__)

EVIDENCE_ONLY_BADGE = "Evidence-only"


def _agent_settings():
    try:
        from config.settings import get_agent_settings
        return get_agent_settings()
    except Exception:
        class _F:
            evidence_signal_limit = 20
            evidence_llm_max_tokens = 400
        return _F()


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


def _normalize_to_evidence_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Convert to canonical EvidenceBundle: nodes [{id, type, label_or_hash, importance, first_seen, last_seen}], edges [{src, dst, type, importance, ts?}], timeline."""
    nodes = []
    for n in bundle.get("nodes") or []:
        nodes.append({
            "id": n.get("id"),
            "type": n.get("type") or n.get("entity_type") or "entity",
            "label_or_hash": bundle.get("entity_displays", {}).get(str(n.get("id")), str(n.get("id", ""))[:8]),
            "importance": n.get("importance", 0.0),
            "first_seen": n.get("first_seen"),
            "last_seen": n.get("last_seen"),
        })
    edges = []
    for e in bundle.get("edges") or []:
        edges.append({
            "src": e.get("src"),
            "dst": e.get("dst"),
            "type": e.get("type") or "related",
            "importance": e.get("importance", 0.0),
            "ts": e.get("ts"),
        })
    timeline = []
    for t in bundle.get("timeline_snippet") or []:
        timeline.append({
            "ts": t.get("ts"),
            "event_type": t.get("event_type"),
            "redacted_text": t.get("text_preview") or t.get("text", ""),
            "entities": t.get("entities", []),
        })
    return {"nodes": nodes, "edges": edges, "timeline": timeline, "entity_ids": list(bundle.get("entity_ids") or [])}


def _what_changed_diff(supabase: Any, household_id: str, signal_id: str, bundle: dict, now: datetime) -> dict[str, Any]:
    """Compare this signal's entity neighborhood vs prior week baseline; novelty score. Persist in explanation.what_changed."""
    entity_ids = set(bundle.get("entity_ids") or [])
    if not supabase or not entity_ids:
        return {"new_entities_count": 0, "new_relationships_count": 0, "novelty_score": 0.0}
    try:
        week_ago = (now - timedelta(days=7)).isoformat()
        r = supabase.table("risk_signals").select("id, explanation").eq("household_id", household_id).lt("ts", week_ago).limit(50).execute()
        prior_entity_ids = set()
        prior_edges = set()
        for row in (r.data or []):
            expl = row.get("explanation") or {}
            sub = expl.get("model_subgraph") or expl.get("subgraph") or {}
            for n in sub.get("nodes") or []:
                prior_entity_ids.add(str(n.get("id", "")))
            for e in sub.get("edges") or []:
                prior_edges.add((str(e.get("src", "")), str(e.get("dst", ""))))
        new_entities = len(entity_ids - prior_entity_ids)
        current_edges = set()
        for e in bundle.get("edges") or []:
            current_edges.add((str(e.get("src", "")), str(e.get("dst", ""))))
        new_edges = len(current_edges - prior_edges)
        novelty = min(1.0, (new_entities / 10.0) * 0.5 + (new_edges / 10.0) * 0.5) if (new_entities or new_edges) else 0.0
        return {"new_entities_count": new_entities, "new_relationships_count": new_edges, "novelty_score": round(novelty, 3)}
    except Exception as e:
        logger.debug("what_changed_diff failed: %s", e)
        return {"new_entities_count": 0, "new_relationships_count": 0, "novelty_score": 0.0}


def _hypotheses_llm(bundle_canonical: dict, allowed_entity_ids: set, allowed_event_ids: set) -> list[dict] | None:
    """Structured hypotheses: title, evidence_points[], uncertainty, next_questions[]. Evidence-only."""
    try:
        from pydantic import BaseModel, Field
        from domain.langchain_utils import get_llm, run_structured_prompt, evidence_only_guard

        class HypothesisItem(BaseModel):
            title: str
            evidence_points: list[str] = Field(default_factory=list)
            uncertainty: str
            next_questions: list[str] = Field(default_factory=list)

        class HypothesesOutput(BaseModel):
            hypotheses: list[HypothesisItem] = Field(default_factory=list)

        llm = get_llm()
        if not llm:
            return None
        prompt = (
            "Given this evidence bundle (nodes, edges, timeline), suggest up to 3 hypotheses that could explain the risk. "
            "Each hypothesis: title, evidence_points (list of short strings referencing ONLY entity ids or event types in the bundle), uncertainty (low/medium/high), next_questions. "
            "Evidence: " + json.dumps(bundle_canonical, default=str)[:2000]
        )
        result = run_structured_prompt(llm, prompt, HypothesesOutput)
        if not result or not result.hypotheses:
            return None
        out = [{"title": h.title, "evidence_points": h.evidence_points, "uncertainty": h.uncertainty, "next_questions": h.next_questions} for h in result.hypotheses[:3]]
        valid, err = evidence_only_guard(out, allowed_entity_ids, allowed_event_ids)
        if not valid:
            return None
        return out
    except Exception as e:
        logger.debug("Hypotheses LLM failed: %s", e)
        return None


def _caregiver_narrative_llm(bundle_canonical: dict, allowed_entity_ids: set) -> dict | None:
    """Structured: headline, summary, key_evidence_bullets[], recommended_next_steps[], confidence_note."""
    try:
        from pydantic import BaseModel, Field
        from domain.langchain_utils import get_llm, run_structured_prompt, evidence_only_guard

        class CaregiverNarrative(BaseModel):
            headline: str
            summary: str
            key_evidence_bullets: list[str] = Field(default_factory=list)
            recommended_next_steps: list[str] = Field(default_factory=list)
            confidence_note: str = ""

        llm = get_llm()
        if not llm:
            return None
        prompt = (
            "Generate a caregiver-facing investigation summary from this evidence bundle only. "
            "Output: headline, summary (2-4 sentences), key_evidence_bullets (3-5), recommended_next_steps (2-4), confidence_note. "
            "Do not reference any entity or fact not in the evidence. Evidence: " + json.dumps(bundle_canonical, default=str)[:2000]
        )
        result = run_structured_prompt(llm, prompt, CaregiverNarrative)
        if not result:
            return None
        out = {"headline": result.headline, "summary": result.summary, "key_evidence_bullets": result.key_evidence_bullets, "recommended_next_steps": result.recommended_next_steps, "confidence_note": result.confidence_note}
        valid, _ = evidence_only_guard(out, allowed_entity_ids, None)
        if not valid:
            return None
        return out
    except Exception as e:
        logger.debug("Caregiver narrative LLM failed: %s", e)
        return None


def _elder_safe_llm(caregiver_summary: str, recommended_steps: list[str]) -> dict | None:
    """Elder-safe: plain_language_summary, do_now_checklist[], reassurance_line. No fear-mongering."""
    try:
        from pydantic import BaseModel, Field
        from domain.langchain_utils import get_llm, run_structured_prompt

        class ElderSafe(BaseModel):
            plain_language_summary: str
            do_now_checklist: list[str] = Field(default_factory=list)
            reassurance_line: str = ""

        llm = get_llm()
        if not llm:
            return None
        prompt = (
            "Convert this caregiver summary into an elder-safe view: one short plain_language_summary (no jargon), "
            "do_now_checklist (2-4 simple actions), reassurance_line (one calming sentence). No fear-mongering; no private details. "
            "Summary: " + caregiver_summary[:500] + ". Suggested steps: " + str(recommended_steps)[:300]
        )
        result = run_structured_prompt(llm, prompt, ElderSafe)
        if not result:
            return None
        return {"plain_language_summary": result.plain_language_summary, "do_now_checklist": result.do_now_checklist, "reassurance_line": result.reassurance_line}
    except Exception as e:
        logger.debug("Elder-safe LLM failed: %s", e)
        return None


def _deterministic_narrative(bundle: dict[str, Any]) -> str:
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
            max_tokens=_agent_settings().evidence_llm_max_tokens,
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


def run_investigation_packager(
    ctx: AgentContext,
    *,
    target: str = "open",
    limit: int = 20,
    risk_signal_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Eight-step Investigation Packager. Returns step_trace, summary_json, status, run_id, artifacts_refs.
    """
    step_trace: list[dict] = []
    started_at = ctx.now.isoformat()
    summary_json: dict[str, Any] = {"headline": "Investigation Packager", "key_metrics": {}, "key_findings": [], "recommended_actions": [], "artifact_refs": {}}
    artifacts_refs: dict[str, Any] = {"risk_signal_ids": [], "summary_ids": []}
    run_id: str | None = None

    if not ctx.supabase:
        step_trace.append({"step": "fetch_signals", "status": "skip", "started_at": started_at, "ended_at": ctx.now.isoformat(), "notes": "no_supabase"})
        summary_json["reason"] = "no_supabase"
        summary_json["key_findings"] = ["Configure Supabase to run packager."]
        run_id = persist_agent_run_ctx(ctx, "evidence_narrative", "completed", step_trace, summary_json, artifacts_refs)
        return {"step_trace": step_trace, "summary_json": summary_json, "status": "ok", "run_id": run_id, "started_at": started_at, "ended_at": ctx.now.isoformat()}

    consent_state = _consent_state(ctx.supabase, ctx.household_id)
    consent_share_text = consent_state.get("share_with_caregiver", True)
    consent_share_entity_canonical = consent_state.get("share_with_caregiver", True)

    # Step 1 — Select signals & fetch evidence
    with step(ctx, step_trace, "fetch_signals", inputs_count=None):
        q = ctx.supabase.table("risk_signals").select("id, explanation, recommended_action").eq("household_id", ctx.household_id).eq("status", target)
        if risk_signal_ids:
            q = q.in_("id", risk_signal_ids[:limit])
        else:
            q = q.order("ts", desc=True).limit(limit)
        r = q.execute()
        signals = r.data or []
        step_trace[-1]["inputs_count"] = limit
        step_trace[-1]["outputs_count"] = len(signals)
        step_trace[-1]["notes"] = f"{len(signals)} signals"

    if not signals:
        summary_json["signals_processed"] = 0
        summary_json["updated"] = 0
        summary_json["key_findings"] = ["No open signals to package."]
        run_id = persist_agent_run_ctx(ctx, "evidence_narrative", "completed", step_trace, summary_json, artifacts_refs)
        return {"step_trace": step_trace, "summary_json": summary_json, "status": "ok", "run_id": run_id, "started_at": started_at, "ended_at": ctx.now.isoformat()}

    # Step 2 — Evidence normalization
    with step(ctx, step_trace, "evidence_normalization"):
        bundles = []
        for s in signals:
            expl = s.get("explanation") or {}
            subgraph = expl.get("model_subgraph") or expl.get("subgraph") or {}
            nodes = subgraph.get("nodes") or []
            edges = subgraph.get("edges") or []
            motifs = expl.get("motif_tags") or []
            if not nodes and not edges and not motifs:
                continue
            bundle = _build_evidence_bundle(s["id"], expl, ctx.supabase, ctx.household_id, consent_share_text, consent_share_entity_canonical)
            canonical = _normalize_to_evidence_bundle(bundle)
            bundles.append({"signal_id": s["id"], "bundle": bundle, "canonical": canonical})
        step_trace[-1]["outputs_count"] = len(bundles)

    # Step 3 — What changed diff (per signal)
    with step(ctx, step_trace, "what_changed_diff"):
        for item in bundles:
            delta = _what_changed_diff(ctx.supabase, ctx.household_id, item["signal_id"], item["bundle"], ctx.now)
            item["what_changed"] = delta
        step_trace[-1]["outputs_count"] = len(bundles)

    # Step 4 — Hypothesis generation (optional LLM)
    with step(ctx, step_trace, "hypothesis_generation"):
        all_entity_ids = set()
        all_event_ids = set()
        for item in bundles:
            all_entity_ids.update(item["bundle"].get("entity_ids") or [])
            for t in (item["bundle"].get("timeline_snippet") or []):
                if t.get("event_type"):
                    all_event_ids.add(str(t.get("event_type", "")))
        hypotheses_list = []
        for item in bundles[:5]:
            hyp = _hypotheses_llm(item["canonical"], all_entity_ids, all_event_ids)
            if hyp:
                item["hypotheses"] = hyp
                hypotheses_list.extend(hyp)
            else:
                item["hypotheses"] = []
        step_trace[-1]["outputs_count"] = len(hypotheses_list)

    # Step 5 — Caregiver narrative (optional LLM)
    with step(ctx, step_trace, "caregiver_narrative"):
        for item in bundles:
            nar = _caregiver_narrative_llm(item["canonical"], set(item["bundle"].get("entity_ids") or []))
            if nar:
                item["caregiver_narrative"] = nar
                narrative_text = nar.get("summary") or nar.get("headline", "")
            else:
                narrative_text = _llm_narrative_if_available(item["bundle"]) or _deterministic_narrative(item["bundle"])
                item["caregiver_narrative"] = {"headline": "Evidence summary", "summary": narrative_text, "key_evidence_bullets": [], "recommended_next_steps": [], "confidence_note": ""}
            item["narrative_text"] = narrative_text
        step_trace[-1]["outputs_count"] = len(bundles)

    # Step 6 — Elder-safe version
    with step(ctx, step_trace, "elder_safe_version"):
        for item in bundles:
            care = item.get("caregiver_narrative") or {}
            elder = _elder_safe_llm(care.get("summary", ""), care.get("recommended_next_steps", []))
            if elder:
                item["elder_safe"] = elder
            else:
                item["elder_safe"] = {"plain_language_summary": "We're reviewing activity for your safety.", "do_now_checklist": ["Check in with your caregiver if anything felt unusual."], "reassurance_line": "You're not alone; we're here to help."}
        step_trace[-1]["outputs_count"] = len(bundles)

    # Step 7 — Persist into risk_signals + summaries + narrative_reports
    narrative_report_id: str | None = None
    with step(ctx, step_trace, "persist_risk_signals_summaries"):
        updated = 0
        for item in bundles:
            signal_id = item["signal_id"]
            expl = next((s.get("explanation") or {} for s in signals if str(s.get("id")) == str(signal_id)), {})
            new_expl = {
                **expl,
                "summary": item.get("narrative_text", ""),
                "narrative": item.get("narrative_text", ""),
                "narrative_evidence_only": True,
                "narrative_elder": item.get("elder_safe"),
                "hypotheses": item.get("hypotheses", []),
                "what_changed": item.get("what_changed", {}),
                "narrative_agent_run": ctx.now.isoformat(),
            }
            if not ctx.dry_run:
                try:
                    ctx.supabase.table("risk_signals").update({"explanation": new_expl}).eq("id", signal_id).eq("household_id", ctx.household_id).execute()
                    updated += 1
                    artifacts_refs["risk_signal_ids"].append(signal_id)
                except Exception as e:
                    logger.warning("Update risk_signal %s failed: %s", signal_id, e)
            else:
                updated += 1
        summary_json["updated"] = updated
        summary_json["signals_processed"] = len(signals)
        summary_json["narrative_preview"] = (bundles[0].get("narrative_text", ""))[:200] if bundles else ""
        summary_json["hypotheses_sample"] = (bundles[0].get("hypotheses", []))[:2] if bundles else []
        summary_json["elder_safe_sample"] = bundles[0].get("elder_safe") if bundles else None
        # Caregiver escalation draft (headline + summary + recommended_next_steps) for visible artifact.
        if bundles:
            care = bundles[0].get("caregiver_narrative") or {}
            summary_json["caregiver_escalation_draft"] = {
                "headline": care.get("headline", ""),
                "summary": care.get("summary", ""),
                "recommended_next_steps": care.get("recommended_next_steps", []),
            }
        period_end = ctx.now.isoformat()
        period_start = (ctx.now - timedelta(days=7)).isoformat()
        summary_id = upsert_summary_ctx(ctx, "weekly_highlight", period_start, period_end, "Investigation Packager run: " + str(updated) + " signals updated.", summary_json)
        if summary_id:
            artifacts_refs["summary_ids"] = [summary_id]
        # Persist narrative_reports row for "View report" (structured output + citations).
        report_signal_ids = [item["signal_id"] for item in bundles]
        report_json = {
            "headline": (bundles[0].get("caregiver_narrative") or {}).get("headline", "Investigation Packager") if bundles else "",
            "narrative_preview": summary_json.get("narrative_preview", ""),
            "reports": [
                {
                    "signal_id": item["signal_id"],
                    "narrative_text": item.get("narrative_text", ""),
                    "caregiver_narrative": item.get("caregiver_narrative"),
                    "elder_safe": item.get("elder_safe"),
                    "hypotheses": item.get("hypotheses", []),
                    "what_changed": item.get("what_changed", {}),
                }
                for item in bundles
            ],
        }
        if not ctx.dry_run and ctx.supabase and report_signal_ids:
            try:
                ins = ctx.supabase.table("narrative_reports").insert({
                    "household_id": ctx.household_id,
                    "agent_run_id": None,
                    "risk_signal_ids": report_signal_ids,
                    "report_json": report_json,
                }).execute()
                if ins.data and len(ins.data) > 0:
                    narrative_report_id = ins.data[0].get("id")
                    artifacts_refs["narrative_report_id"] = narrative_report_id
            except Exception as e:
                logger.warning("Insert narrative_report failed: %s", e)
        step_trace[-1]["outputs_count"] = updated

    # Step 8 — UI integration (notes)
    with step(ctx, step_trace, "ui_integration"):
        summary_json["headline"] = "Investigation Packager complete"
        summary_json["key_metrics"] = {"signals_processed": len(signals), "updated": updated}
        summary_json["key_findings"] = [f"Updated {updated} signal(s) with narrative, hypotheses, and elder-safe view."]
        summary_json["recommended_actions"] = ["View /alerts/[id] for Investigation Packager panel.", "View /elder for elder_safe narrative and do_now checklist."]
        summary_json["artifact_refs"] = artifacts_refs
        step_trace[-1]["notes"] = "/alerts/[id] shows narrative + hypotheses; /elder shows elder_safe"

    run_id = persist_agent_run_ctx(ctx, "evidence_narrative", "completed", step_trace, summary_json, artifacts_refs)
    if run_id and narrative_report_id and not ctx.dry_run and ctx.supabase:
        try:
            ctx.supabase.table("narrative_reports").update({"agent_run_id": run_id}).eq("id", narrative_report_id).execute()
        except Exception as e:
            logger.debug("Update narrative_report agent_run_id: %s", e)
    return {
        "step_trace": step_trace,
        "summary_json": summary_json,
        "status": "ok",
        "started_at": started_at,
        "ended_at": ctx.now.isoformat(),
        "run_id": run_id,
        "artifacts_refs": artifacts_refs,
    }


def run_evidence_narrative_agent(
    household_id: str,
    supabase: Any | None = None,
    dry_run: bool = False,
    *,
    risk_signal_ids: list[str] | None = None,
    risk_signal_id: str | None = None,
) -> dict[str, Any]:
    """Wrapper: build ctx and call run_investigation_packager. Keeps router compatibility."""
    ctx = AgentContext(household_id, supabase, dry_run=dry_run)
    limit = _agent_settings().evidence_signal_limit
    target = "open"
    ids_arg = risk_signal_ids or ([risk_signal_id] if risk_signal_id else None)
    out = run_investigation_packager(ctx, target=target, limit=limit, risk_signal_ids=ids_arg)
    return {"step_trace": out["step_trace"], "summary_json": out["summary_json"], "status": out["status"], "started_at": out["started_at"], "ended_at": out["ended_at"], "run_id": out.get("run_id"), "artifacts_refs": out.get("artifacts_refs")}
