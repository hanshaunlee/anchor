"""Risk signals domain: list, get detail, submit feedback, calibration."""
from datetime import datetime
from uuid import UUID

from supabase import Client

from api.schemas import (
    FeedbackLabel,
    FeedbackSubmit,
    RiskSignalCard,
    RiskSignalDetail,
    RiskSignalListResponse,
    RiskSignalStatus,
)
from .explain_service import build_subgraph_from_explanation


def list_risk_signals(
    household_id: str,
    supabase: Client,
    *,
    status: RiskSignalStatus | None = None,
    severity_min: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> RiskSignalListResponse:
    """List risk signals for household with optional status and severity filters."""
    if not household_id:
        return RiskSignalListResponse(signals=[], total=0)
    q = (
        supabase.table("risk_signals")
        .select("id, ts, signal_type, severity, score, status, explanation", count="exact")
        .eq("household_id", household_id)
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
            model_available=(s.get("explanation") or {}).get("model_available"),
        )
        for s in data
    ]
    return RiskSignalListResponse(signals=signals, total=total)


def get_risk_signal_detail(
    signal_id: UUID,
    household_id: str,
    supabase: Client,
) -> RiskSignalDetail | None:
    """Full risk signal detail including explanation and subgraph. Returns None if not found."""
    r = (
        supabase.table("risk_signals")
        .select("*")
        .eq("id", str(signal_id))
        .eq("household_id", household_id)
        .single()
        .execute()
    )
    if not r.data:
        return None
    s = r.data
    expl = s.get("explanation") or {}
    if "model_available" not in expl:
        expl = {**expl, "model_available": False}
    subgraph = build_subgraph_from_explanation(expl)
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


def submit_feedback(
    signal_id: UUID,
    household_id: str,
    body: FeedbackSubmit,
    user_id: str,
    supabase: Client,
) -> None:
    """
    Store caregiver label (true_positive / false_positive / unsure). On false_positive,
    apply calibration step to household severity threshold.
    Raises ValueError if signal not found or not in household.
    """
    sig = (
        supabase.table("risk_signals")
        .select("id")
        .eq("id", str(signal_id))
        .eq("household_id", household_id)
        .single()
        .execute()
    )
    if not sig.data:
        raise ValueError("Risk signal not found")

    supabase.table("feedback").insert({
        "household_id": household_id,
        "risk_signal_id": str(signal_id),
        "user_id": user_id,
        "label": body.label.value,
        "notes": body.notes,
    }).execute()

    from datetime import timezone
    if body.label in (FeedbackLabel.false_positive, FeedbackLabel.true_positive):
        try:
            from config.settings import get_ml_settings
            ml = get_ml_settings()
            step_fp = ml.calibration_adjust_step
            step_tp = getattr(ml, "calibration_true_positive_step", -0.05)
            cap = getattr(ml, "calibration_adjust_cap", 2.0)
            floor = getattr(ml, "calibration_adjust_floor", -0.5)
        except ImportError:
            step_fp, step_tp, cap, floor = 0.1, -0.05, 2.0, -0.5
        cal = (
            supabase.table("household_calibration")
            .select("severity_threshold_adjust")
            .eq("household_id", household_id)
            .single()
            .execute()
        )
        current = (cal.data.get("severity_threshold_adjust") or 0) if cal.data else 0
        if body.label == FeedbackLabel.false_positive:
            adj = min(cap, current + step_fp)
        else:
            adj = max(floor, current + step_tp)
        supabase.table("household_calibration").upsert(
            {
                "household_id": household_id,
                "severity_threshold_adjust": adj,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="household_id",
        ).execute()
