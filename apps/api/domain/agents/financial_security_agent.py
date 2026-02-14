"""
Financial Security Agent: ordered playbook to protect an elder user.
READ-ONLY external actions: recommend, draft, flag; never move money.
Respects consent_state and household policy; redacts when required.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from domain.graph_service import normalize_events

logger = logging.getLogger(__name__)

# Signal types for financial agent (conventions; no DB enum change)
FINANCIAL_SIGNAL_TYPES = (
    "possible_scam_contact",
    "social_engineering_risk",
    "payment_anomaly",
)


def _pipeline_settings():
    """Pipeline/config settings for thresholds and consent keys (scalable; fallback for demo)."""
    try:
        from config.settings import get_pipeline_settings
        return get_pipeline_settings()
    except ImportError:
        class _Fallback:
            severity_threshold = 4
            consent_share_key = "share_with_caregiver"
            consent_watchlist_key = "watchlist_ok"
            timeline_snippet_max = 6
            persist_score_min = 0.3
            watchlist_score_min = 0.5
        return _Fallback()

# Demo events: synthetic scam scenario (Medicare urgency + share_ssn intent + phone). Shared by API and scripts.
DEMO_EVENTS: list[dict[str, Any]] = [
    {
        "session_id": "s1",
        "device_id": "d1",
        "ts": "2024-01-15T10:00:00Z",
        "seq": 0,
        "event_type": "final_asr",
        "payload": {
            "text": "Someone from Medicare called saying my account is suspended",
            "confidence": 0.9,
            "speaker": {"role": "elder"},
        },
    },
    {
        "session_id": "s1",
        "device_id": "d1",
        "ts": "2024-01-15T10:00:01Z",
        "seq": 1,
        "event_type": "intent",
        "payload": {
            "name": "share_ssn",
            "slots": {"number": "555-1234"},
            "confidence": 0.85,
        },
    },
    {
        "session_id": "s1",
        "device_id": "d1",
        "ts": "2024-01-15T10:00:02Z",
        "seq": 2,
        "event_type": "final_asr",
        "payload": {
            "text": "They said I need to verify immediately",
            "confidence": 0.88,
            "speaker": {"role": "elder"},
        },
    },
]


def get_demo_events() -> list[dict[str, Any]]:
    """Return a copy of the demo events (synthetic scam scenario) for API/scripts."""
    return [dict(e) for e in DEMO_EVENTS]


def _ts_float(ts: Any) -> float:
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return t.timestamp() if t.tzinfo else t.replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            return 0.0
    if hasattr(ts, "timestamp"):
        return ts.timestamp()
    return 0.0


def _hash_for_watchlist(s: str) -> str:
    return hashlib.sha256(s.strip().lower().encode()).hexdigest()[:16]


def _ingest_events(
    household_id: str,
    time_window_days: int,
    supabase: Any | None,
    ingested_events: list[dict] | None,
) -> tuple[list[dict], list[str]]:
    """Task 1a: Pull recent events for household. Return (events, session_ids)."""
    if ingested_events is not None and len(ingested_events) > 0:
        session_ids = list({e.get("session_id") for e in ingested_events if e.get("session_id")})
        return ingested_events, session_ids
    if not supabase:
        return [], []
    from datetime import timedelta
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=time_window_days)
    start_iso = start.isoformat()
    end_iso = end.isoformat()
    sessions_r = (
        supabase.table("sessions")
        .select("id")
        .eq("household_id", household_id)
        .gte("started_at", start_iso)
        .lte("started_at", end_iso)
        .execute()
    )
    session_ids = [s["id"] for s in (sessions_r.data or [])]
    if not session_ids:
        return [], []
    events_list: list[dict] = []
    for sid in session_ids:
        ev_r = (
            supabase.table("events")
            .select("id, session_id, device_id, ts, seq, event_type, payload, payload_version, text_redacted")
            .eq("session_id", sid)
            .order("ts")
            .execute()
        )
        for row in ev_r.data or []:
            row["ts"] = row.get("ts")  # keep as ISO string or datetime
            events_list.append(row)
    events_list.sort(key=lambda x: (_ts_float(x.get("ts")), x.get("seq", 0)))
    return events_list, session_ids


def _detect_risk_patterns(
    utterances: list[dict],
    entities: list[dict],
    mentions: list[dict],
    relationships: list[dict],
    events: list[dict],
    consent_redact: bool,
) -> tuple[list[dict], list[str], list[dict], list[dict], Any]:
    """
    Task 2: Motif/rule layer + optional model layer.
    Returns (risk_scores with signal_type/severity/uncertainty/embedding, motif_tags, timeline_snippet, evidence_subgraph, model_meta).
    model_meta is set when GNN ran (for persisting risk_signal_embeddings); otherwise None.
    """
    entity_id_to_canonical = {e["id"]: e.get("canonical", "") for e in entities}
    motif_tags_global: list[str] = []
    timeline_snippet: list[dict] = []
    try:
        from ml.explainers.motifs import extract_motifs
        motif_tags_global, timeline_snippet = extract_motifs(
            utterances, mentions, entities, relationships, events, entity_id_to_canonical
        )
    except Exception as e:
        logger.warning("Motif extraction failed: %s", e)
    if consent_redact and timeline_snippet:
        for t in timeline_snippet:
            if "text_preview" in t:
                t["text_preview"] = "[redacted]"
    risk_scores: list[dict] = []
    model_meta: Any = None
    if not entities:
        return risk_scores, motif_tags_global, timeline_snippet, [], model_meta
    # Rule layer: motif-based risk
    motif_risk = 0.0
    if motif_tags_global:
        motif_risk = min(0.3 + 0.15 * len(motif_tags_global), 0.95)
    # GNN layer via shared risk scoring service (single contract; no silent placeholders)
    model_scores: dict[int, float] = {}
    model_embeddings: dict[int, list[float]] = {}
    model_available = False
    try:
        from domain.risk_scoring_service import score_risk
        sessions_for_graph = [{"id": s, "started_at": 0} for s in set(u.get("session_id") for u in utterances)]
        if not sessions_for_graph:
            sessions_for_graph = [{"id": "s1", "started_at": 0}]
        response = score_risk(
            "",
            sessions=sessions_for_graph,
            utterances=utterances,
            entities=entities,
            mentions=mentions,
            relationships=relationships,
            devices=[],
            events=events,
        )
        if response.model_available and response.scores:
            model_available = True
            model_meta = response.model_meta
            for s in response.scores:
                model_scores[s.node_index] = s.score
                if s.embedding and isinstance(s.embedding, (list, tuple)) and len(s.embedding) > 0:
                    model_embeddings[s.node_index] = [float(x) for x in s.embedding]
    except Exception as e:
        logger.debug("GNN inference skipped: %s", e)
    for i, _ in enumerate(entities):
        rule_score = motif_risk
        model_score = model_scores.get(i, 0.0)
        combined = 0.6 * rule_score + 0.4 * model_score if model_scores else rule_score
        uncertainty = 0.2 if not model_scores else 0.1
        severity = max(1, min(5, int(1 + combined * 4)))
        signal_type = "possible_scam_contact" if "contact" in str(motif_tags_global).lower() or motif_risk > 0.5 else "social_engineering_risk"
        emb = model_embeddings.get(i)
        risk_scores.append({
            "node_type": "entity",
            "node_index": i,
            "score": round(combined, 4),
            "signal_type": signal_type,
            "severity": severity,
            "uncertainty": uncertainty,
            "motif_rule_score": round(motif_risk, 4),
            "model_score": round(model_scores.get(i, 0.0), 4),
            "model_available": model_available,
            "embedding": emb,
        })
    try:
        from config.settings import get_pipeline_settings
        evidence_score_min = get_pipeline_settings().persist_score_min
    except ImportError:
        evidence_score_min = 0.3
    evidence_subgraph = []
    for r in risk_scores:
        if r.get("score", 0) < evidence_score_min:
            continue
        idx = r.get("node_index", 0)
        if idx < len(entities):
            e = entities[idx]
            evidence_subgraph.append({
                "id": e.get("id"),
                "type": e.get("entity_type", "entity"),
                "label": None if consent_redact else e.get("canonical"),
                "score": r.get("score"),
                "importance": r.get("score", 0),
            })
    timeline_cap = _pipeline_settings().timeline_snippet_max
    return risk_scores, motif_tags_global, timeline_snippet[:timeline_cap], evidence_subgraph, model_meta


def _recommended_action_checklist(
    motif_tags: list[str],
    severity: int,
    has_new_contact: bool,
    has_sensitive_intent: bool,
) -> dict[str, Any]:
    """Task 4: Non-destructive recommended_action with checklist."""
    checklist: list[str] = []
    checklist.append("Call back using saved contact")
    checklist.append("Do not share OTP/codes")
    if has_new_contact or "contact" in str(motif_tags).lower():
        checklist.append("Freeze unknown caller interactions for 60 minutes (recommendation only)")
    checklist.append("Enable bank alerts")
    checklist.append("Change passwords / 2FA")
    checklist.append("Verify payee")
    checklist.append("Review recent transactions and confirm unknown merchants")
    return {
        "checklist": checklist,
        "motif_context": motif_tags,
        "severity": severity,
    }


def _watchlist_synthesis(
    risk_scores: list[dict],
    entities: list[dict],
    motif_tags: list[str],
    score_min: float,
    consent_allows_watchlist: bool,
) -> list[dict]:
    """Task 5: Watchlist items (hashes, keywords, priority, expiry)."""
    if not consent_allows_watchlist:
        return []
    from datetime import timedelta
    expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    items: list[dict] = []
    for r in risk_scores:
        if r.get("score", 0) < score_min:
            continue
        idx = r.get("node_index", 0)
        if idx >= len(entities):
            continue
        e = entities[idx]
        etype = e.get("entity_type", "topic")
        canonical = e.get("canonical", "")
        pattern = {}
        if etype in ("phone", "email", "person"):
            pattern["canonical_hash"] = _hash_for_watchlist(canonical)
            pattern["entity_type"] = etype
        else:
            pattern["entity_type"] = etype
            pattern["canonical_hash"] = _hash_for_watchlist(canonical)
        pattern["score"] = r.get("score")
        items.append({
            "watch_type": "entity_pattern",
            "pattern": pattern,
            "reason": "Financial security agent: " + (motif_tags[0] if motif_tags else "elevated risk"),
            "priority": min(5, 1 + r.get("severity", 1)),
            "expires_at": expires,
        })
    for tag in motif_tags[:3]:
        items.append({
            "watch_type": "keyword",
            "pattern": {"keywords": tag[:50], "topic_hash": _hash_for_watchlist(tag)},
            "reason": "Risky topic pattern",
            "priority": 2,
            "expires_at": expires,
        })
    return items


def _escalation_draft(
    severity: int,
    consent_share: bool,
    threshold: int,
    risk_scores: list[dict],
    motif_tags: list[str],
    low_confidence: bool,
) -> str:
    """Task 6: Draft escalation message only if severity >= threshold and consent; else empty."""
    if low_confidence:
        return ""
    if not consent_share or severity < threshold:
        return ""
    high = [r for r in risk_scores if r.get("severity", 0) >= threshold]
    if not high:
        return ""
    return f"Draft escalation: {len(high)} high-severity financial protection signal(s) for caregiver review. " + ("; ".join(motif_tags[:3]) if motif_tags else "")


def run_financial_security_playbook(
    household_id: str,
    time_window_days: int = 7,
    consent_state: dict | None = None,
    ingested_events: list[dict] | None = None,
    supabase: Any | None = None,
    dry_run: bool = False,
    escalation_severity_threshold: int | None = None,
    persist_score_min: float | None = None,
    watchlist_score_min: float | None = None,
) -> dict[str, Any]:
    """
    Execute the ordered financial protection playbook.
    Returns dict with: risk_signals (payloads), watchlists, logs, run_id, escalation_draft.
    If supabase provided and not dry_run: persists risk_signals, watchlists, agent_runs; broadcasts new alerts.
    Thresholds default to config (get_pipeline_settings) when not passed.
    """
    settings = _pipeline_settings()
    escalation_severity_threshold = escalation_severity_threshold if escalation_severity_threshold is not None else settings.severity_threshold
    persist_score_min = persist_score_min if persist_score_min is not None else settings.persist_score_min
    watchlist_score_min = watchlist_score_min if watchlist_score_min is not None else settings.watchlist_score_min
    consent_state = consent_state or {}
    consent_share = consent_state.get(settings.consent_share_key, True)
    consent_watchlist = consent_state.get(settings.consent_watchlist_key, True)
    consent_redact = not consent_share  # redact sensitive text when not sharing with caregiver
    logs: list[str] = []
    run_id: str | None = None
    started_at = datetime.now(timezone.utc).isoformat()
    step_trace: list[dict] = []

    # 1) Ingest & normalize
    step_trace.append({"step": "ingest", "status": "ok", "started_at": started_at})
    events, session_ids = _ingest_events(household_id, time_window_days, supabase, ingested_events)
    step_trace[-1]["ended_at"] = datetime.now(timezone.utc).isoformat()
    step_trace[-1]["inputs_count"] = 0
    step_trace[-1]["outputs_count"] = len(events)
    step_trace[-1]["notes"] = f"{len(session_ids)} sessions"
    logs.append(f"Financial agent: ingested {len(events)} events, {len(session_ids)} sessions")

    step_trace.append({"step": "normalize", "status": "ok", "started_at": step_trace[-1]["ended_at"]})
    utterances, entities, mentions, relationships = normalize_events(household_id, events)
    step_trace[-1]["ended_at"] = datetime.now(timezone.utc).isoformat()
    step_trace[-1]["outputs_count"] = len(utterances)
    step_trace[-1]["notes"] = f"{len(entities)} entities"
    logs.append(f"Normalized: {len(utterances)} utterances, {len(entities)} entities")

    # 2) Detect risk patterns
    step_trace.append({"step": "detect_risk_patterns", "status": "ok", "started_at": step_trace[-1]["ended_at"]})
    risk_scores, motif_tags, timeline_snippet, evidence_subgraph, model_meta = _detect_risk_patterns(
        utterances, entities, mentions, relationships, events, consent_redact
    )
    step_trace[-1]["ended_at"] = datetime.now(timezone.utc).isoformat()
    step_trace[-1]["outputs_count"] = len(risk_scores)
    step_trace[-1]["notes"] = f"{len(motif_tags)} motif tags"
    logs.append(f"Risk patterns: {len(risk_scores)} scored, {len(motif_tags)} motif tags")

    # Build "what changed vs baseline" summary
    what_changed = []
    if motif_tags:
        what_changed.append("Detected: " + "; ".join(motif_tags[:3]))
    if evidence_subgraph:
        what_changed.append(f"{len(evidence_subgraph)} entities with elevated risk")
    what_changed_summary = " ".join(what_changed) if what_changed else "No significant change vs baseline"

    # 3) Investigation package + 4) Recommendations + 5) Watchlist
    risk_signals_to_persist: list[dict] = []
    for r in risk_scores:
        if r.get("score", 0) < persist_score_min:
            continue
        severity = r.get("severity", 1)
        uncertainty = r.get("uncertainty", 0.2)
        low_confidence = uncertainty > 0.3 or r.get("score", 0) < 0.4
        rec = _recommended_action_checklist(
            motif_tags,
            severity,
            has_new_contact=any("contact" in t.lower() for t in motif_tags),
            has_sensitive_intent=any("sensitive" in t.lower() or "pay" in t.lower() for t in motif_tags),
        )
        explanation = {
            "motif_tags": motif_tags,
            "timeline_snippet": timeline_snippet,
            "subgraph": {"nodes": evidence_subgraph, "edges": []},
            "model_subgraph": {"nodes": evidence_subgraph, "edges": []},
            "what_changed_summary": what_changed_summary,
            "summary": what_changed_summary + f" (entity index {r.get('node_index')}, score {r.get('score', 0):.2f})",
        }
        if consent_redact:
            explanation["redacted"] = True
        escalation_draft = _escalation_draft(
            severity, consent_share, escalation_severity_threshold, risk_scores, motif_tags, low_confidence
        )
        recommended_action = {**rec}
        if escalation_draft and consent_share:
            recommended_action["escalation_draft"] = escalation_draft
        if low_confidence:
            recommended_action["clarification_recommended"] = True
            recommended_action["checklist"] = ["Ask elder to confirm context before escalating"] + rec.get("checklist", [])
        risk_signals_to_persist.append({
            "household_id": household_id,
            "signal_type": r.get("signal_type", "social_engineering_risk"),
            "severity": severity,
            "score": r.get("score", 0),
            "explanation": explanation,
            "recommended_action": recommended_action,
            "status": "open",
            "embedding": r.get("embedding"),
        })

    step_trace.append({"step": "recommendations_watchlist", "status": "ok", "started_at": step_trace[-1]["ended_at"]})
    watchlists_to_persist = _watchlist_synthesis(
        risk_scores, entities, motif_tags, watchlist_score_min, consent_watchlist
    )
    step_trace[-1]["ended_at"] = datetime.now(timezone.utc).isoformat()
    step_trace[-1]["outputs_count"] = len(risk_signals_to_persist) + len(watchlists_to_persist)
    step_trace[-1]["notes"] = f"{len(risk_signals_to_persist)} signals, {len(watchlists_to_persist)} watchlists"
    logs.append(f"Produced {len(risk_signals_to_persist)} risk signals, {len(watchlists_to_persist)} watchlist items")

    # 6) Escalation draft already folded into recommended_action when consent allows
    # 7) Persist + notify (caller broadcasts via /ws/risk_signals using inserted_signals_for_broadcast)
    inserted_signal_ids: list[str] = []
    inserted_signals_for_broadcast: list[dict] = []
    if supabase and not dry_run:
        try:
            run_row = supabase.table("agent_runs").insert({
                "household_id": household_id,
                "agent_name": "financial_security",
                "started_at": started_at,
                "status": "running",
                "step_trace": step_trace,
            }).execute()
            if run_row.data and len(run_row.data) > 0:
                run_id = run_row.data[0].get("id")
            def _embedding_meta():
                """Model version/metadata for risk_signal_embeddings; same contract as worker/jobs.py."""
                try:
                    from config.settings import get_ml_settings
                    ml = get_ml_settings()
                    return getattr(ml, "model_version_tag", None) or "hgt_baseline", getattr(ml, "model_version_tag", None) or "hgt_baseline"
                except ImportError:
                    return "hgt_baseline", "hgt_baseline"
            model_version_tag, default_model_name = _embedding_meta()
            model_name = (model_meta.model_name if model_meta and getattr(model_meta, "model_name", None) else None) or default_model_name
            checkpoint_id = str(model_meta.checkpoint_id) if model_meta and getattr(model_meta, "checkpoint_id", None) else None
            for sig in risk_signals_to_persist:
                row = supabase.table("risk_signals").insert({
                    "household_id": sig["household_id"],
                    "signal_type": sig["signal_type"],
                    "severity": sig["severity"],
                    "score": sig["score"],
                    "explanation": sig["explanation"],
                    "recommended_action": sig["recommended_action"],
                    "status": sig["status"],
                }).execute()
                if row.data and len(row.data) > 0:
                    rid = row.data[0]["id"]
                    ts = row.data[0].get("ts", started_at)
                    inserted_signal_ids.append(rid)
                    inserted_signals_for_broadcast.append({
                        "type": "risk_signal",
                        "id": rid,
                        "household_id": household_id,
                        "ts": ts,
                        "signal_type": sig["signal_type"],
                        "severity": sig["severity"],
                        "score": sig["score"],
                    })
                    # Persist real model-derived embeddings only when present; same schema as worker/jobs.py.
                    emb = sig.get("embedding")
                    if emb and isinstance(emb, (list, tuple)) and len(emb) > 0:
                        vec = [float(x) for x in emb]
                        try:
                            supabase.table("risk_signal_embeddings").upsert({
                                "risk_signal_id": rid,
                                "household_id": household_id,
                                "embedding": vec,
                                "model_version": model_version_tag,
                                "dim": len(vec),
                                "model_name": model_name,
                                "checkpoint_id": checkpoint_id,
                                "has_embedding": True,
                                "meta": {},
                            }, on_conflict="risk_signal_id").execute()
                        except Exception as emb_ex:
                            logger.warning("Financial agent: risk_signal_embeddings upsert failed: %s", emb_ex)
            for wl in watchlists_to_persist:
                supabase.table("watchlists").insert({
                    "household_id": household_id,
                    "watch_type": wl["watch_type"],
                    "pattern": wl["pattern"],
                    "reason": wl["reason"],
                    "priority": wl["priority"],
                    "expires_at": wl.get("expires_at"),
                }).execute()
            if run_id:
                step_trace.append({
                    "step": "persist",
                    "status": "ok",
                    "started_at": step_trace[-1]["ended_at"],
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "outputs_count": len(inserted_signal_ids),
                    "notes": f"inserted {len(inserted_signal_ids)} signals, {len(watchlists_to_persist)} watchlists",
                })
                supabase.table("agent_runs").update({
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "status": "completed",
                    "summary_json": {
                        "risk_signals_created": len(inserted_signal_ids),
                        "watchlists_created": len(watchlists_to_persist),
                        "motif_tags": motif_tags,
                    },
                    "step_trace": step_trace,
                }).eq("id", run_id).execute()
        except Exception as e:
            logger.exception("Financial agent persist failed: %s", e)
            if run_id:
                try:
                    supabase.table("agent_runs").update({
                        "status": "failed",
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "summary_json": {"error": str(e)},
                        "step_trace": step_trace,
                    }).eq("id", run_id).execute()
                except Exception:
                    pass
            logs.append(f"Persist error: {e}")
    else:
        if dry_run:
            logs.append("Dry run: no DB write")

    return {
        "risk_signals": risk_signals_to_persist,
        "watchlists": watchlists_to_persist,
        "logs": logs,
        "run_id": run_id,
        "step_trace": step_trace,
        "inserted_signal_ids": inserted_signal_ids,
        "inserted_signals_for_broadcast": inserted_signals_for_broadcast,
        "session_ids": session_ids,
        "motif_tags": motif_tags,
        "timeline_snippet": timeline_snippet,
    }
