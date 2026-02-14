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
    """Build utterances, entities, mentions, relationships and optionally persist."""
    from ml.graph.builder import GraphBuilder
    builder = GraphBuilder(household_id)
    by_session: dict[str, list] = {}
    for ev in events:
        sid = str(ev.get("session_id", ""))
        if sid not in by_session:
            by_session[sid] = []
        by_session[sid].append(ev)
    for sid, evs in by_session.items():
        builder.process_events(evs, sid, str(evs[0].get("device_id", "")) if evs else "")
    # Persist to Supabase (upsert entities by household_id, entity_type, canonical_hash; insert mentions, relationships)
    entities = builder.get_entity_list()
    mentions = builder.get_mention_list()
    relationships = builder.get_relationship_list()
    # Placeholder: actual DB writes would go here (entity get-or-create, then mentions/relationships)
    result = {"entities": entities, "mentions": mentions, "relationships": relationships, "utterances": builder.get_utterance_list()}
    # Optional: mirror evidence subgraph to Neo4j for visualization (only when NEO4J_URI set)
    try:
        from api.neo4j_sync import sync_evidence_graph_to_neo4j
        if sync_evidence_graph_to_neo4j(household_id, result["entities"], result["relationships"]):
            result["neo4j_synced"] = True
    except ImportError:
        pass
    return result


def run_risk_inference(household_id: str, graph_data: dict[str, list], checkpoint_path: str | None = None) -> list[dict]:
    """Run risk scoring; return list of risk_signal payloads for DB insert."""
    cap = _ml_settings().risk_inference_entity_cap
    entities = graph_data.get("entities", [])[:cap]
    # In production: load model from checkpoint_path, build HeteroData from graph_data, run inference
    risk_scores = []
    for i, _ in enumerate(entities):
        risk_scores.append({
            "household_id": household_id,
            "signal_type": "relational_anomaly",
            "severity": min(5, 1 + int(i % 3)),
            "score": 0.2 + i * 0.05,
            "explanation": {"summary": f"Entity {i} anomaly", "node_index": i},
            "recommended_action": {"action": "review"},
            "status": "open",
        })
    return risk_scores


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
    """If new embedding matches any active embedding_centroid watchlist, add watchlist_match to risk_signal explanation."""
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
            # Update risk_signal explanation to add watchlist_match
            sig_r = (
                supabase_client.table("risk_signals")
                .select("explanation")
                .eq("id", risk_signal_id)
                .eq("household_id", household_id)
                .single()
                .execute()
            )
            if not sig_r.data:
                continue
            expl = dict(sig_r.data.get("explanation") or {})
            expl["watchlist_match"] = {
                "watchlist_id": wl["id"],
                "similarity": round(sim, 4),
                "threshold": threshold,
                "centroid_version": pat.get("model_name") or "hgt_baseline",
            }
            supabase_client.table("risk_signals").update({"explanation": expl}).eq("id", risk_signal_id).eq("household_id", household_id).execute()
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
    return result
