"""Similar incidents from real GNN embeddings only. No synthetic/placeholder vectors."""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from uuid import UUID

from supabase import Client

from api.schemas import RetrievalProvenance, SimilarIncident, SimilarIncidentsResponse


def _similarity_settings():
    try:
        from config.settings import get_agent_settings
        return get_agent_settings()
    except Exception:
        class _F:
            similar_incidents_window_days = 90
            similar_incidents_top_k = 5
        return _F()


def _cos_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-8
    nb = math.sqrt(sum(x * x for x in b)) or 1e-8
    return dot / (na * nb)


def get_similar_incidents(
    signal_id: UUID,
    household_id: str,
    supabase: Client,
    top_k: int | None = None,
    window_days: int | None = None,
) -> SimilarIncidentsResponse:
    """
    Nearest neighbors by real embedding (cosine). Only uses risk_signal_embeddings rows
    with valid embedding; if query signal has no row or has_embedding=false, returns
    available=false, reason=model_not_run. Candidates restricted to same household
    within window_days (from config when not passed).
    """
    cfg = _similarity_settings()
    if top_k is None:
        top_k = cfg.similar_incidents_top_k
    if window_days is None:
        window_days = cfg.similar_incidents_window_days
    # Query embedding for this signal (select has_embedding and provenance columns if present)
    try:
        emb_q = (
            supabase.table("risk_signal_embeddings")
            .select("embedding, has_embedding, model_name, checkpoint_id, dim, created_at")
            .eq("risk_signal_id", str(signal_id))
            .eq("household_id", household_id)
            .limit(1)
            .execute()
        )
    except Exception:
        emb_q = (
            supabase.table("risk_signal_embeddings")
            .select("embedding, has_embedding")
            .eq("risk_signal_id", str(signal_id))
            .eq("household_id", household_id)
            .limit(1)
            .execute()
        )
    data = emb_q.data
    if data is None:
        rows = []
    elif isinstance(data, list):
        rows = data
    else:
        rows = [data]
    if not rows:
        return SimilarIncidentsResponse(available=False, reason="model_not_run", similar=[])
    row = rows[0]
    if row.get("has_embedding") is False:
        return SimilarIncidentsResponse(available=False, reason="model_not_run", similar=[])
    q_emb = row.get("embedding")
    if not q_emb or not isinstance(q_emb, list) or len(q_emb) == 0:
        return SimilarIncidentsResponse(available=False, reason="model_not_run", similar=[])

    # Retrieval provenance when available=True
    prov_ts = None
    if row.get("created_at"):
        try:
            prov_ts = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
        except Exception:
            pass
    retrieval_provenance = RetrievalProvenance(
        model_name=row.get("model_name"),
        checkpoint_id=row.get("checkpoint_id"),
        embedding_dim=row.get("dim"),
        timestamp=prov_ts,
    )

    # Prefer pgvector RPC when extension and embedding_vector exist (migration 008)
    since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    scored: list[tuple[str, float]] = []
    try:
        rpc = supabase.rpc(
            "similar_incidents_by_vector",
            {
                "p_risk_signal_id": str(signal_id),
                "p_household_id": household_id,
                "p_top_k": top_k,
                "p_since": since,
            },
        ).execute()
        if rpc.data and len(rpc.data) > 0:
            scored = [(str(r["risk_signal_id"]), round(float(r["similarity"]), 4)) for r in rpc.data]
    except Exception:
        pass

    # Fallback: JSONB + Python cosine (no pgvector or RPC failed)
    if not scored:
        try:
            cand_q = (
                supabase.table("risk_signal_embeddings")
                .select("risk_signal_id, embedding, has_embedding")
                .eq("household_id", household_id)
                .gte("created_at", since)
                .execute()
            )
        except Exception:
            cand_q = (
                supabase.table("risk_signal_embeddings")
                .select("risk_signal_id, embedding")
                .eq("household_id", household_id)
                .gte("created_at", since)
                .execute()
            )
        candidates = cand_q.data or []
        for r in candidates:
            if r.get("risk_signal_id") == str(signal_id):
                continue
            if r.get("has_embedding") is False:
                continue
            emb = r.get("embedding")
            if not isinstance(emb, list) or len(emb) == 0:
                continue
            sc = _cos_sim(q_emb, emb)
            scored.append((r["risk_signal_id"], sc))
        scored.sort(key=lambda x: -x[1])

    similar: list[SimilarIncident] = []
    for rsid_str, sim in scored[:top_k]:
        rsid = UUID(rsid_str)
        sim_rounded = round(sim, 4)
        # Fetch risk_signal for ts, signal_type, severity, status
        sig_row = (
            supabase.table("risk_signals")
            .select("ts, signal_type, severity, status")
            .eq("id", rsid_str)
            .eq("household_id", household_id)
            .limit(1)
            .execute()
        )
        ts = None
        signal_type = None
        severity = None
        status = None
        if sig_row.data and len(sig_row.data) > 0:
            s = sig_row.data[0]
            if s.get("ts"):
                ts = datetime.fromisoformat(str(s["ts"]).replace("Z", "+00:00"))
            signal_type = s.get("signal_type")
            severity = s.get("severity")
            status = s.get("status")
        # Feedback -> label_outcome
        fb = (
            supabase.table("feedback")
            .select("label")
            .eq("risk_signal_id", rsid_str)
            .limit(1)
            .execute()
        )
        label_outcome = None
        if fb.data and fb.data[0].get("label"):
            lb = fb.data[0]["label"]
            label_outcome = "confirmed_scam" if lb == "true_positive" else "false_positive" if lb == "false_positive" else "open"
        similar.append(
            SimilarIncident(
                risk_signal_id=rsid,
                similarity=sim_rounded,
                score=sim_rounded,
                ts=ts,
                signal_type=signal_type,
                severity=severity,
                status=status,
                label_outcome=label_outcome,
                outcome=label_outcome,
            )
        )
    return SimilarIncidentsResponse(available=True, similar=similar, retrieval_provenance=retrieval_provenance)
