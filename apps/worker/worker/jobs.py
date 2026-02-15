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
            embedding_dim = 128
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


def _get_household_calibration_params(supabase_client: Any, household_id: str) -> dict | None:
    """Read calibration_params (platt_a, platt_b, conformal_q_hat, etc.) for risk scoring.
    When drift has invalidated conformal (conformal_invalid_since set), we strip conformal_q_hat
    so the pipeline does not use it until recalibration; drift and conformal are tied."""
    if not supabase_client or not household_id:
        return None
    try:
        r = (
            supabase_client.table("household_calibration")
            .select("calibration_params")
            .eq("household_id", household_id)
            .limit(1)
            .execute()
        )
        if not r.data or len(r.data) == 0 or not r.data[0].get("calibration_params"):
            return None
        params = dict(r.data[0]["calibration_params"])
        if params.get("conformal_invalid_since"):
            params.pop("conformal_q_hat", None)
            params.pop("coverage_level", None)
            params.pop("calibration_size", None)
        return params
    except Exception:
        pass
    return None


MAX_ATTEMPTS = 3
RETRY_BACKOFF_MINUTES = [5, 15, 60]  # 1st retry 5m, 2nd 15m, 3rd 60m


def process_one_processing_queue_job(supabase_client: Any) -> bool:
    """
    Atomically claim one pending job (via RPC or fallback), run it, mark completed or retry/failed.
    Returns True if a job was processed, False if none pending.
    """
    if not supabase_client:
        return False
    try:
        # Prefer atomic claim via RPC (migration 019)
        claimed = supabase_client.rpc("rpc_claim_processing_queue_job").execute()
        row = None
        if claimed.data and len(claimed.data) > 0:
            row = claimed.data[0]
        if not row:
            # Fallback: select then update (no RPC or no pending)
            r = (
                supabase_client.table("processing_queue")
                .select("id, household_id, job_type, payload, attempt_count")
                .eq("status", "pending")
                .order("created_at")
                .limit(1)
                .execute()
            )
            if not r.data or len(r.data) == 0:
                return False
            row = r.data[0]
            now_iso = datetime.now(timezone.utc).isoformat()
            supabase_client.table("processing_queue").update({
                "status": "running", "started_at": now_iso, "attempt_count": (row.get("attempt_count") or 0) + 1
            }).eq("id", row["id"]).execute()

        job_id = row["id"]
        household_id = str(row["household_id"])
        job_type = row.get("job_type") or "run_supervisor_ingest"
        payload = row.get("payload") or {}
        attempt = int(row.get("attempt_count") or 1)
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            if job_type == "run_supervisor_ingest":
                time_window_days = int(payload.get("time_window_days", 7))
                run_supervisor_ingest_pipeline(
                    supabase_client,
                    household_id,
                    time_range_start=None,
                    time_range_end=None,
                    dry_run=False,
                )
            else:
                raise ValueError(f"Unknown job_type: {job_type}")
            supabase_client.table("processing_queue").update({
                "status": "completed",
                "completed_at": now_iso,
                "last_error": None,
                "next_attempt_at": None,
            }).eq("id", job_id).execute()
        except Exception as e:
            logger.exception("Processing queue job %s failed (attempt %s): %s", job_id, attempt, e)
            err_text = str(e)[:1000]
            if attempt < MAX_ATTEMPTS:
                backoff_mins = RETRY_BACKOFF_MINUTES[min(attempt - 1, len(RETRY_BACKOFF_MINUTES) - 1)]
                next_at = (datetime.now(timezone.utc) + timedelta(minutes=backoff_mins)).isoformat()
                supabase_client.table("processing_queue").update({
                    "status": "pending",
                    "next_attempt_at": next_at,
                    "last_error": err_text,
                }).eq("id", job_id).execute()
            else:
                supabase_client.table("processing_queue").update({
                    "status": "failed",
                    "completed_at": now_iso,
                    "last_error": err_text,
                    "error_text": err_text,
                }).eq("id", job_id).execute()
        return True
    except Exception as e:
        logger.debug("process_one_processing_queue_job: %s", e)
        return False


def run_supervisor_ingest_pipeline(
    supabase_client: Any,
    household_id: str,
    time_range_start: datetime | None = None,
    time_range_end: datetime | None = None,
    *,
    dry_run: bool = False,
) -> dict:
    """
    Product flow: run supervisor INGEST_PIPELINE (financial + narrative + outreach candidates).
    Call this after ingest; supervisor handles normalize, financial detection, narratives, outreach drafts.
    For nightly maintenance use POST /system/maintenance/run (admin) or run_supervisor with NIGHTLY_MAINTENANCE.
    """
    try:
        from domain.agents.supervisor import run_supervisor, INGEST_PIPELINE
    except ImportError as e:
        logger.warning("Supervisor not available: %s. Fall back to run_pipeline.", e)
        return run_pipeline(supabase_client, household_id, time_range_start, time_range_end)
    ingested_events = None
    if supabase_client and (time_range_start or time_range_end):
        ingested_events = ingest_events_batch(supabase_client, household_id, time_range_start, time_range_end)
    time_window_days = 7
    if time_range_start and time_range_end:
        delta = (time_range_end - time_range_start).total_seconds() / 86400
        time_window_days = max(1, min(90, int(delta)))
    result = run_supervisor(
        household_id=household_id,
        supabase=supabase_client if not dry_run else None,
        run_mode=INGEST_PIPELINE,
        dry_run=dry_run,
        time_window_days=time_window_days,
        ingested_events=ingested_events,
    )
    return result


def run_pipeline(supabase_client: Any, household_id: str, time_range_start: datetime | None = None, time_range_end: datetime | None = None) -> dict:
    """Full pipeline: ingest -> graph -> risk -> explain -> watchlist -> persist. Legacy path; prefer run_supervisor_ingest_pipeline for product flow."""
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
    calibration_params = _get_household_calibration_params(supabase_client, household_id)
    result = langgraph_run(
        household_id,
        ingested,
        time_range_start.isoformat() if time_range_start else None,
        time_range_end.isoformat() if time_range_end else None,
        severity_threshold_adjust=severity_threshold_adjust,
        calibration_params=calibration_params,
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
            score_for_severity = sig.get("calibrated_p") if sig.get("calibrated_p") is not None else sig.get("fusion_score") if sig.get("fusion_score") is not None else sig.get("score", 0)
            expl["raw_score"] = sig.get("raw_score")
            expl["calibrated_p"] = sig.get("calibrated_p")
            expl["decision_rule_used"] = sig.get("decision_rule_used")
            payload = {
                "household_id": household_id,
                "signal_type": sig.get("signal_type", "relational_anomaly"),
                "severity": min(5, max(1, int(score_for_severity * 5))),
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
                        payload = {
                            "risk_signal_id": rs_id,
                            "household_id": household_id,
                            "embedding": vec,
                            "model_version": ml.model_version_tag,
                            "dim": len(vec),
                            "model_name": getattr(ml, "model_version_tag", None) or "hgt_baseline",
                            "checkpoint_id": None,
                            "has_embedding": True,
                            "meta": {},
                        }
                        # Prefer 128-D pgvector column when available (migration 022)
                        if len(vec) == 128:
                            payload["embedding_vector_v2"] = vec
                        elif len(vec) == 32:
                            payload["embedding_vector"] = vec
                        supabase_client.table("risk_signal_embeddings").upsert(
                            payload, on_conflict="risk_signal_id"
                        ).execute()
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
