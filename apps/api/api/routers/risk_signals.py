from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import (
    AlertPageBlock,
    AlertPageGating,
    EventListItem,
    FeedbackSubmit,
    PlaybookDetail,
    RiskSignalDetail,
    RiskSignalListResponse,
    RiskSignalPagePayload,
    RiskSignalStatus,
    SimilarIncidentsResponse,
)
from domain.capability_service import get_household_capabilities
from domain.explain_service import run_deep_dive_explainer
from domain.similarity_service import get_similar_incidents
from domain.ingest_service import get_household_id, get_user_role
from domain.risk_service import get_risk_signal_detail, list_risk_signals, submit_feedback

router = APIRouter(prefix="/risk_signals", tags=["risk_signals"])


@router.get("", response_model=RiskSignalListResponse)
def list_risk_signals_route(
    status: RiskSignalStatus | None = Query(None),
    severity_min: int | None = Query(None, alias="severity>=", ge=1, le=5),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    max_age_days: int | None = Query(90, description="Exclude open signals not updated in this many days (phase-out). None = show all."),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """List risk signals. UI: filter by status and minimum severity. Open signals not recently updated are phased out by default."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded. Call POST /households/onboard to create a household.")
    return list_risk_signals(hh_id, supabase, status=status, severity_min=severity_min, limit=limit, offset=offset, max_age_days=max_age_days)


def _consent_allows_share_text(supabase: Client, household_id: str) -> bool:
    """True if latest session consent allows sharing text with caregiver."""
    r = (
        supabase.table("sessions")
        .select("consent_state")
        .eq("household_id", household_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        return True
    consent = r.data[0].get("consent_state") or {}
    return consent.get("share_with_caregiver", True)


@router.get("/{signal_id}", response_model=RiskSignalDetail)
def get_risk_signal_route(
    signal_id: UUID,
    response: Response,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Full detail including explanation_json and evidence pointers (session_ids, event_ids, entity_ids). Redacted when consent disallows sharing text. Cache-friendly: ETag + Cache-Control."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded. Call POST /households/onboard to create a household.")
    consent_ok = _consent_allows_share_text(supabase, hh_id)
    detail = get_risk_signal_detail(signal_id, hh_id, supabase, consent_allows_share_text=consent_ok)
    if not detail:
        raise HTTPException(status_code=404, detail="Risk signal not found")
    row_meta = supabase.table("risk_signals").select("updated_at").eq("id", str(signal_id)).eq("household_id", hh_id).single().execute()
    if row_meta.data and row_meta.data.get("updated_at"):
        etag = f'W/"{signal_id}-{row_meta.data["updated_at"]}"'
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = "private, max-age=10"
    return detail


def _fetch_session_events(supabase: Client, household_id: str, session_id: UUID, limit: int = 10) -> list[EventListItem]:
    """Fetch top N events for a session; verify session belongs to household. Returns [] if not found or not allowed."""
    s = supabase.table("sessions").select("id").eq("id", str(session_id)).eq("household_id", household_id).single().execute()
    if not s.data:
        return []
    q = (
        supabase.table("events")
        .select("id, ts, seq, event_type, payload, text_redacted")
        .eq("session_id", str(session_id))
        .order("ts")
        .limit(limit)
    )
    r = q.execute()
    return [
        EventListItem(
            id=UUID(e["id"]),
            ts=datetime.fromisoformat(e["ts"].replace("Z", "+00:00")),
            seq=e["seq"],
            event_type=e["event_type"],
            payload=e.get("payload") or {},
            text_redacted=e.get("text_redacted", True),
        )
        for e in (r.data or [])
    ]


def _outreach_row_to_dict(row: dict) -> dict:
    """Minimal dict for page payload (no elder redaction; caller can redact if needed)."""
    def _dt(v):
        if v is None:
            return None
        try:
            return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except Exception:
            return None
    return {
        "id": row["id"],
        "household_id": row["household_id"],
        "triggered_by_risk_signal_id": row.get("triggered_by_risk_signal_id"),
        "action_type": row.get("action_type"),
        "channel": row.get("channel"),
        "status": row.get("status"),
        "payload": row.get("payload") or {},
        "error": row.get("error"),
        "created_at": _dt(row.get("created_at")),
        "sent_at": _dt(row.get("sent_at")),
    }


def _compute_page_etag(
    signal_id: UUID,
    signal_updated_at: str | None,
    outbound_actions: list[dict],
    session_events: list[EventListItem],
    similar_available: bool,
    embedding_updated_at: str | None,
) -> str | None:
    """Compute page_etag from all components that affect the page. Any change invalidates cache."""
    import hashlib
    def _ts_str(v):
        if v is None:
            return ""
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return str(v)
    oa_ts = max(
        (_ts_str(a.get("created_at") or a.get("sent_at")) for a in outbound_actions),
        default="",
    )
    ev_ts = max((e.ts.isoformat() for e in session_events), default="")
    parts = [
        str(signal_id),
        signal_updated_at or "",
        oa_ts,
        ev_ts,
        "similar" if similar_available else "no_similar",
        embedding_updated_at or "no_embed",
    ]
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f'W/"page-{h}"'


@router.get("/{signal_id}/page")
def get_risk_signal_page_route(
    signal_id: UUID,
    events_limit: int = Query(10, ge=1, le=50, description="Max session events to return"),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """
    Compound endpoint: one payload for alert detail page.
    Returns composable { detail, page: { session_events, similar, actions, playbook, gating }, page_etag }.
    Cache: React Query caches normally; page_etag invalidates when any component changes.
    Cache-Control: private, max-age=10, stale-while-revalidate=30.
    """
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded. Call POST /households/onboard to create a household.")
    role = get_user_role(supabase, user_id) or "elder"
    consent_ok = _consent_allows_share_text(supabase, hh_id)
    detail = get_risk_signal_detail(signal_id, hh_id, supabase, consent_allows_share_text=consent_ok)
    if not detail:
        raise HTTPException(status_code=404, detail="Risk signal not found")

    similar = get_similar_incidents(signal_id, hh_id, supabase, top_k=5)
    session_events: list[EventListItem] = []
    if detail.session_ids:
        session_events = _fetch_session_events(supabase, hh_id, detail.session_ids[0], limit=events_limit)

    r_oa = supabase.table("outbound_actions").select("*").eq("household_id", hh_id).eq("triggered_by_risk_signal_id", str(signal_id)).order("created_at", desc=True).limit(30).execute()
    outreach_actions = [_outreach_row_to_dict(row) for row in (r_oa.data or [])]
    if role == "elder":
        for a in outreach_actions:
            a["recipient_name"] = None
            a["recipient_contact"] = None
            a["payload"] = {k: v for k, v in (a.get("payload") or {}).items() if k in ("elder_safe_message", "status")}

    playbook: PlaybookDetail | None = None
    pb = supabase.table("action_playbooks").select("*").eq("risk_signal_id", str(signal_id)).eq("household_id", hh_id).order("created_at", desc=True).limit(1).execute()
    if pb.data and len(pb.data) > 0:
        from api.routers.playbooks import _get_playbook_with_tasks
        data = _get_playbook_with_tasks(supabase, str(pb.data[0]["id"]), hh_id, role)
        if data:
            playbook = PlaybookDetail(
                id=UUID(data["id"]),
                household_id=UUID(data["household_id"]),
                risk_signal_id=UUID(data["risk_signal_id"]),
                playbook_type=data["playbook_type"],
                graph=data.get("graph") or {},
                status=data["status"],
                created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
                tasks=data["tasks"],
            )

    caps = get_household_capabilities(supabase, hh_id)
    capabilities_snapshot = {k: v for k, v in caps.items() if k != "household_id"}
    investigation_refresh_allowed = role in ("caregiver", "admin")
    investigation_refresh_reasons = [] if investigation_refresh_allowed else ["role_elder"]

    # Multi-source page_etag: signal, outbound, events, similar/embedding
    row_meta = supabase.table("risk_signals").select("updated_at").eq("id", str(signal_id)).eq("household_id", hh_id).single().execute()
    signal_updated_at = row_meta.data.get("updated_at") if row_meta.data else None
    emb = supabase.table("risk_signal_embeddings").select("created_at").eq("risk_signal_id", str(signal_id)).limit(1).execute()
    embedding_updated_at = None
    if emb.data and len(emb.data) > 0:
        embedding_updated_at = emb.data[0].get("created_at")
    page_etag = _compute_page_etag(
        signal_id, signal_updated_at,
        outreach_actions, session_events, similar.available, embedding_updated_at,
    )

    payload = RiskSignalPagePayload(
        detail=detail,
        page=AlertPageBlock(
            session_events=session_events,
            similar=similar,
            actions=outreach_actions,
            playbook=playbook,
            gating=AlertPageGating(
                investigation_refresh_allowed=investigation_refresh_allowed,
                investigation_refresh_reasons=investigation_refresh_reasons,
                capabilities_snapshot=capabilities_snapshot,
            ),
        ),
        page_etag=page_etag,
    )
    content = payload.model_dump(mode="json")
    # Backward compat: flat keys for existing clients
    content["risk_signal_detail"] = content["detail"]
    content["similar_incidents"] = content["page"]["similar"]
    content["session_events"] = content["page"]["session_events"]
    content["outreach_actions"] = content["page"]["actions"]
    content["playbook"] = content["page"]["playbook"]
    content["capabilities_snapshot"] = content["page"]["gating"]["capabilities_snapshot"]
    content["investigation_refresh_allowed"] = content["page"]["gating"]["investigation_refresh_allowed"]
    content["investigation_refresh_reasons"] = content["page"]["gating"]["investigation_refresh_reasons"]
    response = JSONResponse(content=content)
    if page_etag:
        response.headers["ETag"] = page_etag
    response.headers["Cache-Control"] = "private, max-age=10, stale-while-revalidate=30"
    return response


@router.get("/{signal_id}/similar", response_model=SimilarIncidentsResponse)
def get_similar_incidents_route(
    signal_id: UUID,
    top_k: int = Query(5, ge=1, le=20),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Similar Incidents: retrieve nearest neighbors by embedding (cosine). Shows 3–5 most similar past incidents and outcomes."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded. Call POST /households/onboard to create a household.")
    return get_similar_incidents(signal_id, hh_id, supabase, top_k=top_k)


@router.post("/{signal_id}/explain/deep_dive")
def post_deep_dive_explain_route(
    signal_id: UUID,
    mode: str = Query("pg", description="Explainer mode: pg (PGExplainer) or gnn (GNNExplainer; not yet implemented)"),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Run deep-dive explainer and persist deep_dive_subgraph on the risk signal. mode=pg copies model_subgraph; mode=gnn returns 501."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    if mode not in ("pg", "gnn"):
        raise HTTPException(status_code=400, detail="mode must be 'pg' or 'gnn'")
    try:
        result = run_deep_dive_explainer(signal_id, hh_id, supabase, mode=mode)
        return result
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))


@router.post("/{signal_id}/refresh")
def refresh_alert_route(
    signal_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Re-run supervisor NEW_ALERT for this risk signal (ensure narrative, outreach draft, optional auto-send). Caregiver/admin only."""
    role = get_user_role(supabase, user_id)
    if role not in ("caregiver", "admin"):
        raise HTTPException(status_code=403, detail="Only caregivers or admins can refresh alert")
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    from domain.agents.supervisor import run_supervisor, NEW_ALERT
    result = run_supervisor(
        household_id=hh_id,
        supabase=supabase,
        run_mode=NEW_ALERT,
        dry_run=False,
        risk_signal_id=str(signal_id),
    )
    return {
        "ok": True,
        "supervisor_run_id": result.get("supervisor_run_id"),
        "mode": result.get("mode"),
        "child_run_ids": result.get("child_run_ids"),
        "summary_json": result.get("summary_json"),
        "step_trace": result.get("step_trace"),
    }


@router.post("/{signal_id}/feedback")
def submit_feedback_route(
    signal_id: UUID,
    body: FeedbackSubmit,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Caregiver labels: true_positive / false_positive / unsure. Stored in feedback table."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded. Call POST /households/onboard to create a household.")
    try:
        submit_feedback(signal_id, hh_id, body, user_id, supabase)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True}


@router.get("/{signal_id}/playbook", response_model=PlaybookDetail)
def get_risk_signal_playbook(
    signal_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Shortcut: get playbook for this risk signal (latest active/completed)."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded")
    pb = (
        supabase.table("action_playbooks")
        .select("*")
        .eq("risk_signal_id", str(signal_id))
        .eq("household_id", hh_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not pb.data or len(pb.data) == 0:
        raise HTTPException(status_code=404, detail="No playbook for this signal")
    playbook_id = pb.data[0]["id"]
    role = get_user_role(supabase, user_id) or "elder"
    from api.routers.playbooks import _get_playbook_with_tasks
    data = _get_playbook_with_tasks(supabase, str(playbook_id), hh_id, role)
    if not data:
        raise HTTPException(status_code=404, detail="Playbook not found")
    return PlaybookDetail(
        id=UUID(data["id"]),
        household_id=UUID(data["household_id"]),
        risk_signal_id=UUID(data["risk_signal_id"]),
        playbook_type=data["playbook_type"],
        graph=data.get("graph") or {},
        status=data["status"],
        created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
        updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
        tasks=data["tasks"],
    )