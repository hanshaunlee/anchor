from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import (
    FeedbackLabel,
    FeedbackSubmit,
    RiskSignalCard,
    RiskSignalDetail,
    RiskSignalDetailSubgraph,
    RiskSignalListResponse,
    RiskSignalStatus,
    SimilarIncident,
    SimilarIncidentsResponse,
    SubgraphEdge,
    SubgraphNode,
)

router = APIRouter(prefix="/risk_signals", tags=["risk_signals"])


def _household_id(supabase: Client, user_id: str) -> str | None:
    u = supabase.table("users").select("household_id").eq("id", user_id).single().execute()
    return u.data["household_id"] if u.data else None


@router.get("", response_model=RiskSignalListResponse)
def list_risk_signals(
    status: RiskSignalStatus | None = Query(None),
    severity_min: int | None = Query(None, alias="severity>=", ge=1, le=5),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """List risk signals. UI: filter by status and minimum severity."""
    hh_id = _household_id(supabase, user_id)
    if not hh_id:
        return RiskSignalListResponse(signals=[], total=0)
    q = (
        supabase.table("risk_signals")
        .select("id, ts, signal_type, severity, score, status, explanation", count="exact")
        .eq("household_id", hh_id)
        .order("ts", desc=True)
        .range(offset, offset + limit - 1)
    )
    if status is not None:
        q = q.eq("status", status.value)
    if severity_min is not None:
        q = q.gte("severity", severity_min)
    r = q.execute()
    data = r.data or []
    total = r.count or 0
    signals = [
        RiskSignalCard(
            id=UUID(s["id"]),
            ts=datetime.fromisoformat(s["ts"].replace("Z", "+00:00")),
            signal_type=s["signal_type"],
            severity=s["severity"],
            score=s["score"],
            status=RiskSignalStatus(s["status"]),
            summary=(s.get("explanation") or {}).get("summary"),
        )
        for s in data
    ]
    return RiskSignalListResponse(signals=signals, total=total)


@router.get("/{signal_id}", response_model=RiskSignalDetail)
def get_risk_signal(
    signal_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Full detail including explanation_json and evidence pointers (session_ids, event_ids, entity_ids)."""
    hh_id = _household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    r = (
        supabase.table("risk_signals")
        .select("*")
        .eq("id", str(signal_id))
        .eq("household_id", hh_id)
        .single()
        .execute()
    )
    if not r.data:
        raise HTTPException(status_code=404, detail="Risk signal not found")
    s = r.data
    expl = s.get("explanation") or {}
    subgraph_data = expl.get("subgraph") or expl.get("model_subgraph") or {}
    nodes = [SubgraphNode(id=str(n.get("id", "")), type=n.get("type", ""), label=n.get("label"), score=n.get("score")) for n in subgraph_data.get("nodes", [])]
    edges = [SubgraphEdge(src=e["src"], dst=e["dst"], type=e.get("type", ""), weight=e.get("weight"), rank=e.get("rank")) for e in subgraph_data.get("edges", [])]
    subgraph = RiskSignalDetailSubgraph(nodes=nodes, edges=edges) if (nodes or edges) else None
    return RiskSignalDetail(
        id=UUID(s["id"]),
        household_id=UUID(s["household_id"]),
        ts=datetime.fromisoformat(s["ts"].replace("Z", "+00:00")),
        signal_type=s["signal_type"],
        severity=s["severity"],
        score=s["score"],
        status=RiskSignalStatus(s["status"]),
        explanation=expl,
        recommended_action=s.get("recommended_action") or {},
        subgraph=subgraph,
        session_ids=[UUID(x) for x in expl.get("session_ids", [])],
        event_ids=[UUID(x) for x in expl.get("event_ids", [])],
        entity_ids=[UUID(x) for x in expl.get("entity_ids", [])],
    )


@router.get("/{signal_id}/similar", response_model=SimilarIncidentsResponse)
def get_similar_incidents(
    signal_id: UUID,
    top_k: int = Query(5, ge=1, le=20),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Similar Incidents: retrieve nearest neighbors by embedding (cosine). Shows 3–5 most similar past incidents and outcomes."""
    hh_id = _household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    # Get current signal's embedding
    emb_row = (
        supabase.table("risk_signal_embeddings")
        .select("embedding")
        .eq("risk_signal_id", str(signal_id))
        .eq("household_id", hh_id)
        .single()
        .execute()
    )
    if not emb_row.data:
        return SimilarIncidentsResponse(similar=[])
    q_emb = emb_row.data.get("embedding")
    if not q_emb or not isinstance(q_emb, list):
        return SimilarIncidentsResponse(similar=[])
    # Load all embeddings for household
    all_rows = (
        supabase.table("risk_signal_embeddings")
        .select("risk_signal_id, embedding")
        .eq("household_id", hh_id)
        .execute()
    )
    rows = all_rows.data or []
    # Cosine similarity in Python
    import math
    def cos_sim(a, b):
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-8
        nb = math.sqrt(sum(x * x for x in b)) or 1e-8
        return dot / (na * nb)
    scored = []
    for r in rows:
        if r["risk_signal_id"] == str(signal_id):
            continue
        emb = r.get("embedding")
        if not isinstance(emb, list):
            continue
        sc = cos_sim(q_emb, emb)
        scored.append((UUID(r["risk_signal_id"]), sc))
    scored.sort(key=lambda x: -x[1])
    # Fetch outcome from feedback for each
    similar = []
    for rsid, score in scored[:top_k]:
        fb = supabase.table("feedback").select("label").eq("risk_signal_id", str(rsid)).limit(1).execute()
        outcome = None
        if fb.data and fb.data[0].get("label"):
            outcome = "confirmed_scam" if fb.data[0]["label"] == "true_positive" else "false_positive" if fb.data[0]["label"] == "false_positive" else "open"
        sig = supabase.table("risk_signals").select("ts").eq("id", str(rsid)).single().execute()
        ts = datetime.fromisoformat(sig.data["ts"].replace("Z", "+00:00")) if sig.data and sig.data.get("ts") else None
        similar.append(SimilarIncident(risk_signal_id=rsid, score=round(score, 4), outcome=outcome, ts=ts))
    return SimilarIncidentsResponse(similar=similar)


@router.post("/{signal_id}/feedback")
def submit_feedback(
    signal_id: UUID,
    body: FeedbackSubmit,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Caregiver labels: true_positive / false_positive / unsure. Stored in feedback table."""
    hh_id = _household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    # Verify signal belongs to household
    sig = supabase.table("risk_signals").select("id").eq("id", str(signal_id)).eq("household_id", hh_id).single().execute()
    if not sig.data:
        raise HTTPException(status_code=404, detail="Risk signal not found")
    supabase.table("feedback").insert({
        "household_id": hh_id,
        "risk_signal_id": str(signal_id),
        "user_id": user_id,
        "label": body.label.value,
        "notes": body.notes,
    }).execute()
    # HITL: false positive -> raise threshold slightly for this household (calibration)
    if body.label == FeedbackLabel.false_positive:
        from datetime import datetime, timezone
        try:
            from config.settings import get_ml_settings
            step = get_ml_settings().calibration_adjust_step
        except ImportError:
            step = 0.1
        cal = supabase.table("household_calibration").select("severity_threshold_adjust").eq("household_id", hh_id).single().execute()
        adj = (cal.data.get("severity_threshold_adjust") or 0) + step
        supabase.table("household_calibration").upsert(
            {"household_id": hh_id, "severity_threshold_adjust": adj, "updated_at": datetime.now(timezone.utc).isoformat()},
            on_conflict="household_id",
        ).execute()
    return {"ok": True}
