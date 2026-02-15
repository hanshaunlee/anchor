"""
Supervisor Orchestrator: unified product flow (Investigation + Actions + Model Health).
Modes: INGEST_PIPELINE (primary), NEW_ALERT, NIGHTLY_MAINTENANCE, ADMIN_BENCH.
Scoring uses calibrated_p + fusion_score when available; conformal (q_hat) influences escalation consistently.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from domain.agents.base import AgentContext, persist_agent_run_ctx, step
from domain.graph_service import normalize_events

logger = logging.getLogger(__name__)

SUPERVISOR_SLUG = "supervisor"
RunMode = str  # "INGEST_PIPELINE" | "NEW_ALERT" | "NIGHTLY_MAINTENANCE" | "ADMIN_BENCH"
INGEST_PIPELINE = "INGEST_PIPELINE"
NEW_ALERT = "NEW_ALERT"
NIGHTLY_MAINTENANCE = "NIGHTLY_MAINTENANCE"
ADMIN_BENCH = "ADMIN_BENCH"


def _pipeline_settings():
    try:
        from config.settings import get_pipeline_settings
        return get_pipeline_settings()
    except ImportError:
        class _F:
            severity_threshold = 4
            persist_score_min = 0.3
        return _F()


def _fetch_household_context(supabase: Any, household_id: str) -> dict[str, Any]:
    """Fetch calibration_params, consent snapshot, capabilities."""
    out: dict[str, Any] = {"calibration_params": None, "consent_state": {}, "capabilities": {}}
    if not supabase:
        return out
    try:
        cal = supabase.table("household_calibration").select("calibration_params, meta").eq("household_id", household_id).limit(1).execute()
        if cal.data and len(cal.data) > 0:
            out["calibration_params"] = (cal.data[0].get("calibration_params") or {}) if isinstance(cal.data[0].get("calibration_params"), dict) else None
        sess = supabase.table("sessions").select("consent_state").eq("household_id", household_id).order("started_at", desc=True).limit(1).execute()
        if sess.data and len(sess.data) > 0:
            out["consent_state"] = sess.data[0].get("consent_state") or {}
        try:
            from domain.capability_service import get_household_capabilities
            out["capabilities"] = get_household_capabilities(supabase, household_id)
        except Exception:
            pass
    except Exception as e:
        logger.debug("Fetch household context failed: %s", e)
    return out


def _escalation_triggered(
    calibrated_p: float | None,
    fusion_score: float | None,
    severity: int,
    conformal_q_hat: float | None,
    escalation_threshold: int,
) -> bool:
    """True if we should create outreach candidate. Conformal: escalate only if (1 - calibrated_p) >= q_hat when q_hat exists."""
    if conformal_q_hat is not None and calibrated_p is not None:
        return (1.0 - calibrated_p) >= conformal_q_hat
    score = calibrated_p if calibrated_p is not None else fusion_score
    if score is None:
        return severity >= escalation_threshold
    return severity >= escalation_threshold and (score >= 0.5 or severity >= escalation_threshold)


def run_supervisor(
    household_id: str,
    supabase: Any | None = None,
    *,
    run_mode: str = INGEST_PIPELINE,
    dry_run: bool = False,
    risk_signal_id: str | None = None,
    time_window_days: int = 7,
    force_agents: list[str] | None = None,
    env: str = "prod",
    ingested_events: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Run supervisor in the given mode. Returns SupervisorRunResult-shaped dict.
    """
    ctx = AgentContext(household_id=household_id, supabase=supabase, dry_run=dry_run)
    step_trace: list[dict] = []
    started_at = ctx.now.isoformat()
    result: dict[str, Any] = {
        "supervisor_run_id": None,
        "mode": run_mode,
        "child_run_ids": {},
        "created_signal_ids": [],
        "updated_signal_ids": [],
        "created_watchlist_ids": [],
        "outreach_candidates": [],
        "summary_json": {"counts": {}, "decisions": [], "thresholds_used": {}, "warnings": []},
        "step_trace": step_trace,
        "warnings": [],
    }

    if run_mode == NIGHTLY_MAINTENANCE:
        with step(ctx, step_trace, "nightly_maintenance"):
            from domain.agents.model_health_agent import run_model_health_agent
            mh = run_model_health_agent(household_id, supabase=supabase, dry_run=dry_run, env=env)
            result["child_run_ids"]["model_health"] = mh.get("run_id")
            result["summary_json"] = mh.get("summary_json") or result["summary_json"]
            step_trace[-1]["outputs_count"] = 1
        result["step_trace"] = step_trace
        result["supervisor_run_id"] = persist_agent_run_ctx(
            ctx, "supervisor", "completed", step_trace,
            {"mode": run_mode, "child_run_ids": result["child_run_ids"], "summary_json": result["summary_json"]},
        )
        return result

    if run_mode == ADMIN_BENCH:
        with step(ctx, step_trace, "admin_bench"):
            for agent_slug in (force_agents or []):
                try:
                    from domain.agents.registry import get_agent_by_slug
                    spec = get_agent_by_slug(agent_slug)
                    if spec and spec.get("run_entrypoint"):
                        mod_name, fn_name = spec["run_entrypoint"].rsplit(":", 1)
                        import importlib
                        mod = importlib.import_module(mod_name)
                        fn = getattr(mod, fn_name)
                        bench_result = fn(household_id, supabase=supabase, dry_run=dry_run) if fn_name == "run_graph_drift_agent" else fn(household_id, supabase=supabase, dry_run=dry_run)
                        result["child_run_ids"][agent_slug] = bench_result.get("run_id")
                except Exception as e:
                    logger.exception("Admin bench agent %s failed: %s", agent_slug, e)
                    result["warnings"].append(f"{agent_slug}: {e}")
            step_trace[-1]["outputs_count"] = len(force_agents or [])
        result["step_trace"] = step_trace
        result["supervisor_run_id"] = persist_agent_run_ctx(
            ctx, "supervisor", "completed", step_trace,
            {"mode": run_mode, "child_run_ids": result["child_run_ids"], "warnings": result["warnings"]},
        )
        return result

    if run_mode == NEW_ALERT and risk_signal_id:
        # Ensure narrative exists, re-evaluate conformal + outreach draft, optionally auto_send
        context = _fetch_household_context(ctx.supabase, household_id)
        consent = context.get("consent_state") or ctx.consent_state
        caps = context.get("capabilities") or {}
        auto_send = caps.get("auto_send_outreach", False)
        consent_outbound = consent.get("outbound_contact_ok", True)

        with step(ctx, step_trace, "ensure_narrative"):
            try:
                from domain.agents.evidence_narrative_agent import run_evidence_narrative_agent
                nar = run_evidence_narrative_agent(household_id, supabase=supabase, dry_run=dry_run, risk_signal_ids=[risk_signal_id])
                result["child_run_ids"]["narrative"] = nar.get("run_id")
            except Exception as e:
                logger.warning("Narrative for NEW_ALERT failed: %s", e)
            step_trace[-1]["outputs_count"] = 1

        with step(ctx, step_trace, "outreach_draft_or_send"):
            # Create or update outreach draft; if auto_send and consent, execute send (call caregiver_outreach send path)
            try:
                from domain.agents.caregiver_outreach_agent import run_caregiver_outreach_agent
                send_now = auto_send and consent_outbound and not dry_run
                outreach_result = run_caregiver_outreach_agent(
                    household_id, supabase, risk_signal_id=risk_signal_id,
                    dry_run=not send_now if send_now else dry_run,
                    consent_state=consent, user_role="caregiver",
                )
                result["child_run_ids"]["outreach"] = outreach_result.get("run_id")
                result["summary_json"]["outreach_sent"] = (outreach_result.get("summary_json") or {}).get("sent", False)
            except Exception as e:
                logger.warning("Outreach for NEW_ALERT failed: %s", e)
            step_trace[-1]["outputs_count"] = 1

        result["step_trace"] = step_trace
        result["supervisor_run_id"] = persist_agent_run_ctx(
            ctx, "supervisor", "completed", step_trace,
            {"mode": run_mode, "risk_signal_id": risk_signal_id, "child_run_ids": result["child_run_ids"], "summary_json": result["summary_json"]},
        )
        return result

    # INGEST_PIPELINE
    settings = _pipeline_settings()
    context = _fetch_household_context(ctx.supabase, household_id)
    calibration_params = context.get("calibration_params")
    consent_state = context.get("consent_state") or ctx.consent_state

    with step(ctx, step_trace, "load_context"):
        step_trace[-1]["outputs_count"] = 1
        step_trace[-1]["notes"] = f"calibration_present={bool(calibration_params)}"

    events = ingested_events
    if events is None and ctx.supabase:
        start = ctx.now - timedelta(days=time_window_days)
        end = ctx.now
        sess_r = (
            ctx.supabase.table("sessions")
            .select("id")
            .eq("household_id", household_id)
            .gte("started_at", start.isoformat())
            .lte("started_at", end.isoformat())
            .execute()
        )
        session_ids = [s["id"] for s in (sess_r.data or [])]
        events = []
        for sid in session_ids[:50]:
            ev_r = ctx.supabase.table("events").select("id, session_id, device_id, ts, seq, event_type, payload").eq("session_id", sid).order("ts").limit(500).execute()
            events.extend(ev_r.data or [])

    entities, relationships = [], []
    with step(ctx, step_trace, "normalize_events"):
        if not events:
            step_trace[-1]["notes"] = "no_events"
            result["warnings"].append("no_events")
            result["summary_json"]["counts"] = {"new_signals": 0, "updated_signals": 0, "watchlists": 0, "outreach_candidates": 0}
            result["supervisor_run_id"] = persist_agent_run_ctx(ctx, "supervisor", "completed", step_trace, result["summary_json"])
            return result
        utterances, entities, mentions, relationships = normalize_events(household_id, events)
        step_trace[-1]["outputs_count"] = len(utterances)
        step_trace[-1]["notes"] = f"{len(entities)} entities"

    with step(ctx, step_trace, "run_financial_detection"):
        from domain.agents.financial_security_agent import run_financial_security_playbook
        financial_result = run_financial_security_playbook(
            household_id=household_id,
            time_window_days=time_window_days,
            consent_state=consent_state,
            ingested_events=events,
            supabase=supabase if not dry_run else None,
            dry_run=dry_run,
            escalation_severity_threshold=getattr(settings, "severity_threshold", 4),
            persist_score_min=getattr(settings, "persist_score_min", 0.3),
            calibration_params=calibration_params,
        )
        result["child_run_ids"]["financial"] = financial_result.get("run_id")
        result["created_signal_ids"] = financial_result.get("inserted_signal_ids") or []
        result["created_watchlist_ids"] = []  # financial doesn't return watchlist ids; count from summary
        result["summary_json"]["counts"]["new_signals"] = len(result["created_signal_ids"])
        result["summary_json"]["counts"]["watchlists"] = len(financial_result.get("watchlists") or [])
        step_trace[-1]["outputs_count"] = len(financial_result.get("risk_signals") or [])
        step_trace[-1]["notes"] = f"signals={len(result['created_signal_ids'])}"

    # Ensure narratives for new/open signals (idempotent: narrative agent updates existing)
    signal_ids_for_narrative = result["created_signal_ids"][:]
    if ctx.supabase and not dry_run:
        open_r = ctx.supabase.table("risk_signals").select("id").eq("household_id", household_id).eq("status", "open").limit(50).execute()
        existing_open = [str(r["id"]) for r in (open_r.data or [])]
        for sid in existing_open:
            if sid not in signal_ids_for_narrative:
                signal_ids_for_narrative.append(sid)
    if signal_ids_for_narrative:
        with step(ctx, step_trace, "ensure_narratives"):
            try:
                from domain.agents.evidence_narrative_agent import run_evidence_narrative_agent
                nar = run_evidence_narrative_agent(household_id, supabase=supabase, dry_run=dry_run, risk_signal_ids=signal_ids_for_narrative[:20])
                result["child_run_ids"]["narrative"] = nar.get("run_id")
                step_trace[-1]["outputs_count"] = len(signal_ids_for_narrative[:20])
            except Exception as e:
                logger.warning("Ensure narratives failed: %s", e)
                step_trace[-1]["status"] = "error"
                step_trace[-1]["error"] = str(e)

    # Structural motifs from first signal (for optional_ring_discovery)
    structural_motifs: list[dict] = []
    for sig in financial_result.get("risk_signals") or []:
        structural_motifs = (sig.get("explanation") or {}).get("structural_motifs") or []
        if structural_motifs:
            break

    # Outreach candidates: for each signal above threshold, conformal condition -> create draft only
    conformal_q_hat = None
    if calibration_params and isinstance(calibration_params, dict):
        conformal_q_hat = calibration_params.get("conformal_q_hat")
    escalation_threshold = getattr(settings, "severity_threshold", 4)
    risk_signals = financial_result.get("risk_signals") or []
    outreach_candidates: list[dict] = []
    for sig in risk_signals:
        severity = sig.get("severity", 0)
        expl = sig.get("explanation") or {}
        cal_p = expl.get("calibrated_p")
        fusion = expl.get("fusion_score")
        if not _escalation_triggered(cal_p, fusion, severity, conformal_q_hat, escalation_threshold):
            continue
        if not consent_state.get("outbound_contact_ok", True):
            continue
        outreach_candidates.append({
            "risk_signal_id": None,  # not persisted yet in dry_run; would be from inserted_signal_ids
            "severity": severity,
            "calibrated_p": cal_p,
            "fusion_score": fusion,
            "decision_rule_used": expl.get("decision_rule_used"),
        })
    result["outreach_candidates"] = outreach_candidates
    result["summary_json"]["counts"]["outreach_candidates"] = len(outreach_candidates)

    # Persist draft outbound_actions when not dry_run (link by created_signal_ids by index if matching)
    if supabase and not dry_run and outreach_candidates and result["created_signal_ids"]:
        try:
            for i, cand in enumerate(outreach_candidates[:10]):
                if i >= len(result["created_signal_ids"]):
                    break
                sig_id = result["created_signal_ids"][i]
                row = {
                    "household_id": household_id,
                    "triggered_by_risk_signal_id": sig_id,
                    "action_type": "caregiver_notify",
                    "channel": "sms",
                    "payload": {"evidence_bundle_summary": True, "calibrated_score_context": cand},
                    "status": "queued",
                }
                # Optional columns (migration 015); omit if not present to stay backward compatible
                try:
                    row["conformal_triggered"] = conformal_q_hat is not None
                    row["calibrated_p_at_send"] = cand.get("calibrated_p")
                    row["fusion_score_at_send"] = cand.get("fusion_score")
                    row["decision_rule_used"] = cand.get("decision_rule_used")
                except Exception:
                    pass
                ins = supabase.table("outbound_actions").insert(row).execute()
                if ins.data and len(ins.data) > 0:
                    pass  # draft created
        except Exception as e:
            logger.warning("Create outreach drafts failed: %s", e)
            result["warnings"].append(str(e))

    # Optional ring discovery: only if sufficient relationships or structural motifs indicate star/bridge
    ring_triggered = False
    if events and len(entities or []) >= 2 and len(relationships or []) >= 1:
        motif_types = " ".join(str(m.get("pattern_type", "")) for m in (structural_motifs or [])).lower()
        if "star" in motif_types or "bridge" in motif_types or "triadic" in motif_types or len(relationships or []) >= 3:
            ring_triggered = True
    ring_result: dict[str, Any] = {}
    if ring_triggered and not dry_run and supabase:
        with step(ctx, step_trace, "optional_ring_discovery"):
            try:
                from domain.agents.ring_discovery_agent import run_ring_discovery_agent
                ring_result = run_ring_discovery_agent(household_id, supabase=supabase, neo4j_available=False, dry_run=False)
                result["child_run_ids"]["ring"] = ring_result.get("run_id")
                step_trace[-1]["outputs_count"] = 1
                step_trace[-1]["notes"] = "ring_discovery_run"
            except Exception as e:
                logger.warning("Optional ring discovery failed: %s", e)
                step_trace[-1]["status"] = "error"
                step_trace[-1]["error"] = str(e)

    with step(ctx, step_trace, "finalize_and_broadcast"):
        if result["created_signal_ids"] and not dry_run:
            try:
                from api.broadcast import broadcast_risk_signal
                for sid in result["created_signal_ids"][:20]:
                    broadcast_risk_signal({"type": "risk_signal", "id": sid, "household_id": household_id})
            except ImportError:
                pass
        step_trace[-1]["notes"] = "broadcast_done"

    # Recurring contacts: another weight for watchlist (returning callers / repeat contacts)
    recurring_result: dict[str, Any] = {}
    if events and not dry_run and ctx.supabase:
        with step(ctx, step_trace, "recurring_contacts"):
            try:
                from domain.agents.recurring_contacts_agent import run_recurring_contacts_agent
                recurring_result = run_recurring_contacts_agent(
                    household_id,
                    ctx.supabase,
                    events=events,
                    time_window_days=time_window_days,
                    dry_run=False,
                )
                result["child_run_ids"]["recurring_contacts"] = recurring_result.get("run_id")
                step_trace[-1]["outputs_count"] = len(recurring_result.get("watchlist_items") or [])
                step_trace[-1]["notes"] = "recurring_contacts_run"
            except Exception as e:
                logger.warning("Recurring contacts agent failed: %s", e)
                step_trace[-1]["status"] = "error"
                step_trace[-1]["error"] = str(e)

    result["summary_json"]["thresholds_used"] = {"escalation_severity": escalation_threshold, "conformal_q_hat": conformal_q_hat}
    result["step_trace"] = step_trace
    result["supervisor_run_id"] = persist_agent_run_ctx(
        ctx, "supervisor", "completed", step_trace,
        {"mode": run_mode, "child_run_ids": result["child_run_ids"], "summary_json": result["summary_json"], "created_signal_ids": result["created_signal_ids"], "outreach_candidates_count": len(outreach_candidates)},
    )
    # Apply watchlist batch for this run so protection UI shows one batch per investigation
    batch_id = result.get("supervisor_run_id")
    watchlist_items = list(financial_result.get("watchlists") or [])
    watchlist_items.extend(ring_result.get("watchlist_items") or [])
    watchlist_items.extend(recurring_result.get("watchlist_items") or [])
    # Seed device protections section when capability is enabled (so "Device protections" isn't blank)
    caps = context.get("capabilities") or {}
    if caps.get("device_policy_push_enabled", True):
        watchlist_items.append({
            "watch_type": "high_risk_mode",
            "pattern": {"high_risk_mode": "enabled"},
            "reason": "Device policy push enabled; alerts can be sent to device.",
            "priority": 1,
            "expires_at": None,
        })
    if batch_id and ctx.supabase and not dry_run:
        try:
            from domain.watchlists.service import upsert_watchlist_batch
            ids = upsert_watchlist_batch(
                ctx.supabase,
                household_id,
                batch_id,
                watchlist_items,
                source_agent="supervisor",
                source_run_id=batch_id,
            )
            result["created_watchlist_ids"] = ids
            if result["summary_json"].get("counts") is not None:
                result["summary_json"]["counts"]["watchlists"] = len(ids)
        except Exception as e:
            logger.warning("Supervisor watchlist batch failed: %s", e)
    return result
