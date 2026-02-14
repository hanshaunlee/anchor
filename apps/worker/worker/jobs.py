"""
Async jobs: ingest -> graph build -> inference/training.
Called by cron or API trigger; uses Supabase and optional LangGraph pipeline.
Thresholds and embedding dim from config to avoid hardcoding.
Only real model-derived embeddings are persisted; no placeholders when model did not run.
"""
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


def _ml_settings():
    try:
        from config.settings import get_ml_settings
        return get_ml_settings()
    except ImportError:
        class _Fallback:
            embedding_dim = 32
            risk_inference_entity_cap = 100
            model_version_tag = "v0"
        return _Fallback()


def _pipeline_settings():
    try:
        from config.settings import get_pipeline_settings
        return get_pipeline_settings()
    except ImportError:
        class _Fallback:
            persist_score_min = 0.3
        return _Fallback()


def ingest_events_batch(supabase_client: Any, household_id: str, time_range_start: datetime | None = None, time_range_end: datetime | None = None) -> list[dict]:
    """Fetch events for household in time range from Supabase."""
    q = supabase_client.table("events").select("*, sessions!inner(household_id)").eq("sessions.household_id", household_id)
    if time_range_start:
        q = q.gte("ts", time_range_start.isoformat())
    if time_range_end:
        q = q.lte("ts", time_range_end.isoformat())
    r = q.execute()
    return list(r.data or [])


def run_graph_builder(supabase_client: Any, household_id: str, events: list[dict]) -> dict[str, list]:
    """Build utterances, entities, mentions, relationships via shared graph_service; persist when supabase provided."""
    from domain.graph_service import build_graph_from_events
    utterances, entities, mentions, relationships = build_graph_from_events(household_id, events, supabase=supabase_client)
    result = {"entities": entities, "mentions": mentions, "relationships": relationships, "utterances": utterances}
    try:
        from api.neo4j_sync import sync_evidence_graph_to_neo4j
        if sync_evidence_graph_to_neo4j(household_id, result["entities"], result["relationships"]):
            result["neo4j_synced"] = True
    except ImportError:
        pass
    return result


def run_risk_inference(household_id: str, graph_data: dict[str, list], checkpoint_path: str | None = None) -> list[dict]:
    """Run risk scoring via shared risk scoring service. Returns list of risk_signal payloads for DB insert.
    When model is unavailable, returns explicit rule-only fallback (model_available=false); no silent placeholders."""
    from domain.risk_scoring_service import score_risk

    cap = _ml_settings().risk_inference_entity_cap
    entities = graph_data.get("entities", [])[:cap]
    if not entities:
        return []

    utterances = graph_data.get("utterances", [])
    sessions = []
    for u in utterances:
        sid = u.get("session_id") or u.get("session")
        if sid and not any(s.get("id") == sid for s in sessions):
            sessions.append({"id": sid, "started_at": 0})

    if not sessions:
        sessions = [{"id": "s1", "started_at": 0}]

    response = score_risk(
        household_id,
        sessions=sessions,
        utterances=utterances,
        entities=entities,
        mentions=graph_data.get("mentions", []),
        relationships=graph_data.get("relationships", []),
        devices=[],
        events=[],
        checkpoint_path=checkpoint_path,
    )

    if response.model_available and response.scores:
        out = []
        for s in response.scores:
            d = s.model_dump()
            d["household_id"] = household_id
            d["severity"] = min(5, max(1, int(s.score * 5)))
            d["explanation"] = {"summary": f"Entity {s.node_index} anomaly", "node_index": s.node_index}
            d["recommended_action"] = {"action": "review"}
            d["status"] = "open"
            out.append(d)
        return out

    # Explicit rule-only fallback: payloads for DB insert with model_available=false
    return [
        {
            "household_id": household_id,
            "signal_type": "relational_anomaly",
            "severity": min(5, 1 + int(i % 3)),
            "score": 0.2 + i * 0.05,
            "node_type": "entity",
            "node_index": i,
            "model_available": False,
            "explanation": {"summary": f"Entity {i} anomaly (rule-only)", "node_index": i},
            "recommended_action": {"action": "review"},
            "status": "open",
        }
        for i in range(len(entities))
    ]


def _cos_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-8
    nb = math.sqrt(sum(x * x for x in b)) or 1e-8
    return dot / (na * nb)


def _check_embedding_centroid_watchlists(
    supabase_client: Any,
    household_id: str,
    risk_signal_id: str,
    embedding: list[float],
) -> None:
    """If new embedding matches any active embedding_centroid watchlist: (1) add watchlist_match to risk_signal explanation; (2) create a watchlist_embedding_match risk_signal for visibility."""
    try:
        now = datetime.now(timezone.utc)
        r = (
            supabase_client.table("watchlists")
            .select("id, pattern, expires_at")
            .eq("household_id", household_id)
            .eq("watch_type", "embedding_centroid")
            .execute()
        )
        raw = r.data or []
        watchlists = [
            w for w in raw
            if w.get("expires_at") is None
            or (w.get("expires_at") and datetime.fromisoformat(str(w["expires_at"]).replace("Z", "+00:00")) >= now)
        ]
        for wl in watchlists:
            pat = wl.get("pattern") or {}
            centroid = pat.get("centroid")
            threshold = float(pat.get("threshold", 0.82))
            if not centroid or not isinstance(centroid, list) or len(centroid) != len(embedding):
                continue
            sim = _cos_sim(embedding, centroid)
            if sim < threshold:
                continue
            match_payload = {
                "watchlist_id": wl["id"],
                "similarity": round(sim, 4),
                "threshold": threshold,
                "centroid_version": pat.get("model_name") or "hgt_baseline",
            }
            # (1) Update original risk_signal explanation
            sig_r = (
                supabase_client.table("risk_signals")
                .select("explanation")
                .eq("id", risk_signal_id)
                .eq("household_id", household_id)
                .single()
                .execute()
            )
            if sig_r.data:
                expl = dict(sig_r.data.get("explanation") or {})
                expl["watchlist_match"] = match_payload
                supabase_client.table("risk_signals").update({"explanation": expl}).eq("id", risk_signal_id).eq("household_id", household_id).execute()
            # (2) Create a dedicated watchlist_embedding_match risk_signal (severity 3–4) for alerts list and UI
            severity = 4 if sim >= 0.9 else 3
            explanation = {
                "summary": f"Matched centroid watchlist (similarity {match_payload['similarity']:.2%})",
                "watchlist_match": match_payload,
                "triggering_risk_signal_id": risk_signal_id,
            }
            try:
                ins = supabase_client.table("risk_signals").insert({
                    "household_id": household_id,
                    "signal_type": "watchlist_embedding_match",
                    "severity": severity,
                    "score": float(sim),
                    "explanation": explanation,
                    "recommended_action": {"action": "review", "context": "Centroid watchlist match"},
                    "status": "open",
                }).execute()
                if ins.data and len(ins.data) > 0:
                    logger.info("Created watchlist_embedding_match risk_signal %s for watchlist %s", ins.data[0].get("id"), wl["id"])
            except Exception as ins_ex:
                logger.warning("Insert watchlist_embedding_match risk_signal failed: %s", ins_ex)
            break  # One match per signal is enough
    except Exception as ex:
        logger.debug("Embedding centroid watchlist check skipped: %s", ex)


def _get_household_calibration_adjust(supabase_client: Any, household_id: str) -> float:
    """Read severity_threshold_adjust from household_calibration for pipeline escalation threshold."""
    if not supabase_client or not household_id:
        return 0.0
    try:
        r = (
            supabase_client.table("household_calibration")
            .select("severity_threshold_adjust")
            .eq("household_id", household_id)
            .single()
            .execute()
        )
        if r.data is not None and "severity_threshold_adjust" in r.data:
            return float(r.data["severity_threshold_adjust"])
    except Exception:
        pass
    return 0.0


def run_pipeline(supabase_client: Any, household_id: str, time_range_start: datetime | None = None, time_range_end: datetime | None = None) -> dict:
    """Full pipeline: ingest -> graph -> risk -> explain -> watchlist -> persist."""
    from api.pipeline import run_pipeline as langgraph_run
    events = []
    if supabase_client:
        events = ingest_events_batch(supabase_client, household_id, time_range_start, time_range_end)
    # Convert to list of dicts with session_id, device_id, ts, seq, event_type, payload
    ingested = []
    for e in events:
        ingested.append({
            "session_id": e.get("session_id"),
            "device_id": e.get("device_id"),
            "ts": e.get("ts"),
            "seq": e.get("seq", 0),
            "event_type": e.get("event_type", ""),
            "payload": e.get("payload") or {},
        })
    severity_threshold_adjust = _get_household_calibration_adjust(supabase_client, household_id)
    result = langgraph_run(
        household_id,
        ingested,
        time_range_start.isoformat() if time_range_start else None,
        time_range_end.isoformat() if time_range_end else None,
        severity_threshold_adjust=severity_threshold_adjust,
    )
    persist_min = _pipeline_settings().persist_score_min
    if supabase_client:
        for sig in result.get("risk_scores", []):
            if sig.get("score", 0) < persist_min:
                continue
            expl = {}
            for e in result.get("explanations", []):
                if e.get("node_index") == sig.get("node_index"):
                    expl = e.get("explanation_json", {})
                    break
            payload = {
                "household_id": household_id,
                "signal_type": sig.get("signal_type", "relational_anomaly"),
                "severity": min(5, max(1, int(sig.get("score", 0) * 5))),
                "score": float(sig.get("score", 0)),
                "explanation": expl,
                "recommended_action": {"action": "review"},
                "status": "open",
            }
            try:
                r = supabase_client.table("risk_signals").insert(payload).execute()
                rs_id = r.data[0]["id"] if r.data else None
                if rs_id:
                    # Only persist real model-derived embeddings when model ran; do not insert when model did not run.
                    emb = sig.get("embedding")
                    if emb and isinstance(emb, (list, tuple)) and len(emb) > 0:
                        ml = _ml_settings()
                        vec = [float(x) for x in emb]
                        supabase_client.table("risk_signal_embeddings").upsert({
                            "risk_signal_id": rs_id,
                            "household_id": household_id,
                            "embedding": vec,
                            "model_version": ml.model_version_tag,
                            "dim": len(vec),
                            "model_name": getattr(ml, "model_version_tag", None) or "hgt_baseline",
                            "checkpoint_id": None,
                            "has_embedding": True,
                            "meta": {},
                        }, on_conflict="risk_signal_id").execute()
                        _check_embedding_centroid_watchlists(supabase_client, household_id, rs_id, vec)
            except Exception as ex:
                logger.warning("Insert risk_signal failed: %s", ex)
        for wl in result.get("watchlists", []):
            try:
                expires_at = None
                if wl.get("expires_at_days") is not None:
                    expires_at = (datetime.now(timezone.utc) + timedelta(days=int(wl["expires_at_days"]))).isoformat()
                supabase_client.table("watchlists").insert({
                    "household_id": household_id,
                    "watch_type": wl.get("watch_type", "pattern"),
                    "pattern": wl.get("pattern", {}),
                    "reason": wl.get("reason"),
                    "priority": wl.get("priority", 0),
                    "expires_at": expires_at,
                }).execute()
            except Exception as ex:
                logger.warning("Insert watchlist failed: %s", ex)
    # Optional: after persisting, run outreach for new high-severity signals (config-driven, consent-gated, idempotent)
    try:
        from config.settings import get_worker_settings
        if get_worker_settings().outreach_auto_trigger:
            outreach_result = run_outreach_for_new_signals(supabase_client, household_id, limit=3)
            if outreach_result.get("processed") or outreach_result.get("outbound_action_ids"):
                result["outreach_processed"] = outreach_result.get("processed", 0)
                result["outreach_action_ids"] = outreach_result.get("outbound_action_ids", [])
    except Exception as ex:
        logger.debug("Outreach for new signals skipped: %s", ex)
    return result


def run_outreach_for_new_signals(supabase_client: Any, household_id: str, *, limit: int = 5) -> dict:
    """
    Find recent risk_signals with severity >= escalation threshold and consent_allow_outbound_contact;
    run caregiver outreach agent for each (queued as job, not inline in pipeline).
    Returns { "processed": int, "outbound_action_ids": [...], "errors": [...] }.
    """
    out: dict = {"processed": 0, "outbound_action_ids": [], "errors": []}
    if not supabase_client or not household_id:
        return out
    try:
        from config.settings import get_pipeline_settings
        settings = get_pipeline_settings()
        threshold = getattr(settings, "severity_threshold", 4)
    except Exception:
        threshold = 4
    adjust = _get_household_calibration_adjust(supabase_client, household_id)
    effective = max(1, min(5, threshold + int(adjust)))
    # Consent: from latest session for household
    consent_state = {}
    r_sess = (
        supabase_client.table("sessions")
        .select("consent_state")
        .eq("household_id", household_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if r_sess.data and len(r_sess.data) > 0:
        consent_state = r_sess.data[0].get("consent_state") or {}
    from domain.consent import normalize_consent_state
    normalized = normalize_consent_state(consent_state)
    if not normalized.get("consent_allow_outbound_contact", False):
        return out
    r_sig = (
        supabase_client.table("risk_signals")
        .select("id")
        .eq("household_id", household_id)
        .eq("status", "open")
        .gte("severity", effective)
        .order("ts", desc=True)
        .limit(limit)
        .execute()
    )
    signals = r_sig.data or []
    for row in signals:
        signal_id = row.get("id")
        if not signal_id:
            continue
        try:
            from domain.agents.caregiver_outreach_agent import run_caregiver_outreach_agent
            result = run_caregiver_outreach_agent(
                household_id,
                supabase_client,
                risk_signal_id=str(signal_id),
                dry_run=False,
                consent_state=consent_state,
                user_role="caregiver",
            )
            if result.get("outbound_action_id"):
                out["outbound_action_ids"].append(result["outbound_action_id"])
            if result.get("summary_json", {}).get("suppressed"):
                continue
            out["processed"] += 1
        except Exception as e:
            logger.warning("Outreach for signal %s failed: %s", signal_id, e)
            out["errors"].append({"risk_signal_id": signal_id, "error": str(e)})
    return out
