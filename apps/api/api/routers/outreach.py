"""Outbound caregiver outreach: trigger and list outbound_actions. Caregiver/admin only for trigger; elder sees elder_safe only."""
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from supabase import Client

from api.deps import get_supabase, require_user, require_caregiver_or_admin
from api.schemas import OutboundActionResponse, OutreachActionRequest
from domain.agents.caregiver_outreach_agent import run_caregiver_outreach_agent
from domain.ingest_service import get_household_id, get_user_role

router = APIRouter(prefix="/actions", tags=["outreach"])


def _user_readable_message(raw: Any) -> str | None:
    """Return message only if it looks like user-facing text; never expose internal notes."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    if any(x in s.lower() for x in ("signal loaded", "consent_allow", "contacts=", "step_trace")):
        return None
    return s


def _redact_for_elder(row: dict) -> dict:
    """Return only elder-safe fields for elder role."""
    payload = row.get("payload") or {}
    return {
        "id": row.get("id"),
        "household_id": row.get("household_id"),
        "triggered_by_risk_signal_id": row.get("triggered_by_risk_signal_id"),
        "action_type": row.get("action_type"),
        "channel": row.get("channel"),
        "recipient_name": None,
        "recipient_contact": None,
        "payload": {"elder_safe_message": payload.get("elder_safe_message"), "status": row.get("status")},
        "status": row.get("status"),
        "provider": row.get("provider"),
        "provider_message_id": None,
        "error": None,
        "created_at": row.get("created_at"),
        "sent_at": row.get("sent_at"),
        "delivered_at": row.get("delivered_at"),
    }


def _row_to_response(row: dict, redact_contact: bool = False) -> dict:
    if redact_contact:
        row = _redact_for_elder(row)
    def _dt(v: Any) -> datetime | None:
        if v is None:
            return None
        s = str(v).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None
    out = {
        "id": row["id"],
        "household_id": row["household_id"],
        "triggered_by_risk_signal_id": row.get("triggered_by_risk_signal_id"),
        "action_type": row.get("action_type"),
        "channel": row.get("channel"),
        "recipient_name": row.get("recipient_name"),
        "recipient_contact": row.get("recipient_contact"),
        "recipient_contact_last4": row.get("recipient_contact_last4"),
        "payload": row.get("payload") or {},
        "status": row.get("status"),
        "provider": row.get("provider"),
        "provider_message_id": row.get("provider_message_id"),
        "error": row.get("error"),
        "created_at": _dt(row.get("created_at")),
        "sent_at": _dt(row.get("sent_at")),
        "delivered_at": _dt(row.get("delivered_at")),
    }
    return out


@router.post("/outreach/preview")
def post_outreach_preview(
    body: OutreachActionRequest,
    user_id: str = Depends(require_caregiver_or_admin),
    supabase: Client = Depends(get_supabase),
):
    """Preview outreach message drafts (caregiver_full, elder_safe, evidence bundle, calibrated score context). Caregiver/admin only."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    role = get_user_role(supabase, user_id) or "caregiver"
    consent_state = {}
    r = supabase.table("sessions").select("consent_state").eq("household_id", hh_id).order("started_at", desc=True).limit(1).execute()
    if r.data and len(r.data) > 0:
        consent_state = r.data[0].get("consent_state") or {}
    result = run_caregiver_outreach_agent(
        hh_id, supabase, risk_signal_id=str(body.risk_signal_id), dry_run=True,
        consent_state=consent_state, user_role=role,
    )
    sj = result.get("summary_json") or {}
    return {
        "ok": True,
        "preview": {
            "caregiver_full": _user_readable_message(sj.get("caregiver_message") or sj.get("channel")),
            "elder_safe": sj.get("elder_safe_message"),
            "evidence_bundle_summary": sj.get("evidence_bundle_summary"),
            "calibrated_score_context": sj.get("calibrated_score_context"),
            "step_trace": result.get("step_trace"),
        },
        "suppressed": sj.get("suppressed", False),
    }


@router.post("/outreach/send")
def post_outreach_send(
    body: OutreachActionRequest,
    user_id: str = Depends(require_caregiver_or_admin),
    supabase: Client = Depends(get_supabase),
):
    """Execute outreach send for a risk signal. Caregiver/admin only; requires outbound consent."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    role = get_user_role(supabase, user_id) or "caregiver"
    consent_state = {}
    r = supabase.table("sessions").select("consent_state").eq("household_id", hh_id).order("started_at", desc=True).limit(1).execute()
    if r.data and len(r.data) > 0:
        consent_state = r.data[0].get("consent_state") or {}
    result = run_caregiver_outreach_agent(
        hh_id, supabase, risk_signal_id=str(body.risk_signal_id), dry_run=False,
        consent_state=consent_state, user_role=role,
    )
    outbound_action = None
    if result.get("outbound_action_id"):
        row = supabase.table("outbound_actions").select("*").eq("id", result["outbound_action_id"]).single().execute()
        if row.data:
            outbound_action = _row_to_response(row.data, redact_contact=False)
    return {
        "ok": True,
        "outbound_action": outbound_action,
        "agent_run_id": result.get("run_id"),
        "sent": (result.get("summary_json") or {}).get("sent", False),
        "suppressed": (result.get("summary_json") or {}).get("suppressed", False),
    }


@router.post("/outreach")
def post_outreach(
    body: OutreachActionRequest,
    user_id: str = Depends(require_caregiver_or_admin),
    supabase: Client = Depends(get_supabase),
):
    """
    Trigger caregiver outreach for a risk signal. Caregiver/admin only.
    Returns outbound_action, agent_run_id, preview (when dry_run).
    """
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    role = get_user_role(supabase, user_id) or "caregiver"
    consent_state = {}
    r = (
        supabase.table("sessions")
        .select("consent_state")
        .eq("household_id", hh_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if r.data and len(r.data) > 0:
        consent_state = r.data[0].get("consent_state") or {}
    result = run_caregiver_outreach_agent(
        hh_id,
        supabase,
        risk_signal_id=str(body.risk_signal_id),
        dry_run=body.dry_run,
        consent_state=consent_state,
        user_role=role,
    )
    outbound_action = None
    if result.get("outbound_action_id") and not body.dry_run:
        row = (
            supabase.table("outbound_actions")
            .select("*")
            .eq("id", result["outbound_action_id"])
            .single()
            .execute()
        )
        if row.data:
            outbound_action = _row_to_response(row.data, redact_contact=False)
    preview = None
    if body.dry_run:
        sj = result.get("summary_json") or {}
        preview = {
            "caregiver_message": _user_readable_message(sj.get("caregiver_message") or sj.get("channel")),
            "elder_safe_message": sj.get("elder_safe_message"),
            "step_trace": result.get("step_trace"),
        }
    return {
        "ok": True,
        "outbound_action": outbound_action,
        "agent_run_id": result.get("run_id"),
        "preview": preview,
        "suppressed": (result.get("summary_json") or {}).get("suppressed", False),
        "sent": (result.get("summary_json") or {}).get("sent", False),
    }


@router.get("/outreach")
def list_outreach(
    household_id: UUID | None = Query(None, description="Filter by household (must match yours)"),
    risk_signal_id: UUID | None = Query(None, description="Filter by risk signal (history for one alert)"),
    limit: int = Query(20, ge=1, le=100),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """List recent outbound actions for the current user's household. Optional risk_signal_id filter."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    if household_id and str(household_id) != hh_id:
        raise HTTPException(status_code=403, detail="household_id must match your household.")
    role = get_user_role(supabase, user_id) or "elder"
    q = supabase.table("outbound_actions").select("*").eq("household_id", hh_id)
    if risk_signal_id:
        q = q.eq("triggered_by_risk_signal_id", str(risk_signal_id))
    r = q.order("created_at", desc=True).limit(limit).execute()
    redact = role == "elder"
    return {"actions": [_row_to_response(row, redact_contact=redact) for row in (r.data or [])]}


@router.get("/outreach/candidates")
def get_outreach_candidates(
    user_id: str = Depends(require_caregiver_or_admin),
    supabase: Client = Depends(get_supabase),
):
    """
    List outreach candidates for the household: queued outbound_actions plus risk_signal context.
    Returns blocking reasons (consent, caregiver contact, already sent). Caregiver/admin only.
    """
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    role = get_user_role(supabase, user_id) or "caregiver"
    # Merge household consent defaults (Settings) with latest session so "Allow outbound contact" is respected
    consent_state = {}
    r_hh = supabase.table("household_consent_defaults").select("*").eq("household_id", hh_id).limit(1).execute()
    if r_hh.data and len(r_hh.data) > 0:
        row = r_hh.data[0]
        consent_state = {
            "allow_outbound_contact": row.get("allow_outbound_contact", False),
            "share_with_caregiver": row.get("share_with_caregiver", True),
        }
    r_sess = (
        supabase.table("sessions")
        .select("consent_state")
        .eq("household_id", hh_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if r_sess.data and len(r_sess.data) > 0:
        sess = r_sess.data[0].get("consent_state") or {}
        for k in ("outbound_contact_ok", "allow_outbound_contact", "consent_allow_outbound_contact", "share_with_caregiver"):
            if k in sess and sess[k] is not None:
                consent_state[k] = sess[k]
    outbound_ok = consent_state.get("outbound_contact_ok", consent_state.get("consent_allow_outbound_contact", consent_state.get("allow_outbound_contact", False)))
    share_ok = consent_state.get("share_with_caregiver", True)
    missing_consent = []
    if not outbound_ok:
        missing_consent.append("outbound_contact_ok")
    if not share_ok:
        missing_consent.append("share_with_caregiver")
    contacts_r = (
        supabase.table("caregiver_contacts")
        .select("id")
        .eq("household_id", hh_id)
        .limit(1)
        .execute()
    )
    caregiver_contact_present = bool(contacts_r.data and len(contacts_r.data) > 0)
    q = (
        supabase.table("outbound_actions")
        .select("id, triggered_by_risk_signal_id, status, channel, created_at, payload")
        .eq("household_id", hh_id)
        .eq("status", "queued")
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    rows = q.data or []
    signal_ids = [r["triggered_by_risk_signal_id"] for r in rows if r.get("triggered_by_risk_signal_id")]
    signals_by_id = {}
    if signal_ids:
        sig_r = (
            supabase.table("risk_signals")
            .select("id, severity, signal_type, created_at, status")
            .eq("household_id", hh_id)
            .in_("id", list(set(signal_ids)))
            .execute()
        )
        for s in (sig_r.data or []):
            signals_by_id[s["id"]] = s
    sent_signal_ids = set()
    sent_r = (
        supabase.table("outbound_actions")
        .select("triggered_by_risk_signal_id")
        .eq("household_id", hh_id)
        .in_("status", ["sent", "delivered"])
        .execute()
    )
    for row in (sent_r.data or []):
        if row.get("triggered_by_risk_signal_id"):
            sent_signal_ids.add(row["triggered_by_risk_signal_id"])
    candidates = []
    for row in rows:
        rsid = row.get("triggered_by_risk_signal_id")
        if not rsid:
            continue
        sig = signals_by_id.get(rsid) or {}
        blocking_reasons = []
        if not outbound_ok:
            blocking_reasons.append("consent_outbound")
        if not share_ok:
            blocking_reasons.append("consent_share")
        if not caregiver_contact_present:
            blocking_reasons.append("no_caregiver_contact")
        if rsid in sent_signal_ids:
            blocking_reasons.append("already_sent")
        payload = row.get("payload") or {}
        cal = payload.get("calibrated_score_context") or {}
        candidates.append({
            "risk_signal_id": rsid,
            "outbound_action_id": row.get("id"),
            "severity": sig.get("severity"),
            "signal_type": sig.get("signal_type"),
            "created_at": row.get("created_at") or sig.get("created_at"),
            "candidate_reason": cal.get("decision_rule_used") or "severity and conformal trigger",
            "consent_ok": outbound_ok and share_ok,
            "missing_consent_keys": missing_consent if not (outbound_ok and share_ok) else [],
            "caregiver_contact_present": caregiver_contact_present,
            "blocking_reasons": blocking_reasons,
            "draft_available": True,
        })
    return {"candidates": candidates}


@router.get("/outreach/summary")
def get_outreach_summary(
    user_id: str = Depends(require_caregiver_or_admin),
    supabase: Client = Depends(get_supabase),
):
    """Counts (sent, suppressed, failed) and recent outbound actions for Agents page. Caregiver/admin only."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    all_r = (
        supabase.table("outbound_actions")
        .select("status")
        .eq("household_id", hh_id)
        .execute()
    )
    counts = {"sent": 0, "suppressed": 0, "failed": 0, "queued": 0, "delivered": 0}
    for row in (all_r.data or []):
        s = (row.get("status") or "").lower()
        if s in counts:
            counts[s] += 1
    recent_r = (
        supabase.table("outbound_actions")
        .select("id, status, created_at, sent_at, error, triggered_by_risk_signal_id, channel, recipient_contact_last4")
        .eq("household_id", hh_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    data = recent_r.data or []
    return {
        "counts": counts,
        "recent": [
            {
                "id": x.get("id"),
                "status": x.get("status"),
                "created_at": x.get("created_at"),
                "sent_at": x.get("sent_at"),
                "error": x.get("error"),
                "triggered_by_risk_signal_id": x.get("triggered_by_risk_signal_id"),
                "channel": x.get("channel"),
                "recipient_contact_last4": x.get("recipient_contact_last4"),
            }
            for x in data
        ],
    }


@router.get("/outreach/{action_id}")
def get_outreach(
    action_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Get one outbound action. Caregiver/admin see full; elder sees elder_safe_message only."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    role = get_user_role(supabase, user_id) or "elder"
    r = (
        supabase.table("outbound_actions")
        .select("*")
        .eq("id", str(action_id))
        .eq("household_id", hh_id)
        .single()
        .execute()
    )
    if not r.data:
        raise HTTPException(status_code=404, detail="Outbound action not found.")
    redact = role == "elder"
    return _row_to_response(r.data, redact_contact=redact)
