"""Protection: unified watchlists, rings, reports for the Protection page."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supabase import Client

from api.deps import get_supabase, require_user
from domain.ingest_service import get_household_id
from domain.watchlist_service import list_watchlists
from domain.watchlists.service import list_active_watchlist_items

router = APIRouter(prefix="/protection", tags=["protection"])


# --- Response models ---
class WatchlistItemDisplay(BaseModel):
    id: str
    category: str
    type: str
    display_label: str | None
    display_value: str | None
    explanation: str | None
    priority: int
    score: float | None
    source_agent: str | None
    evidence_signal_ids: list[str] = Field(default_factory=list)


class WatchlistSummary(BaseModel):
    total: int
    by_category: dict[str, int]
    items: list[WatchlistItemDisplay] = Field(default_factory=list)


class RingSummary(BaseModel):
    id: str
    household_id: str
    created_at: str
    updated_at: str
    score: float
    summary_label: str | None = None
    summary_text: str | None = None
    members_count: int = 0
    meta: dict = Field(default_factory=dict)


class ReportSummary(BaseModel):
    kind: str
    last_run_at: str | None
    last_run_id: str | None
    summary: str | None
    status: str | None


class ProtectionOverview(BaseModel):
    watchlist_summary: WatchlistSummary
    rings_summary: list[RingSummary] = Field(default_factory=list)
    reports_summary: list[ReportSummary] = Field(default_factory=list)
    last_updated_at: str | None = None
    data_freshness: dict = Field(default_factory=dict)


class ProtectionSummary(BaseModel):
    """GET /protection/summary: counts + short previews for dashboard cards."""
    updated_at: str | None = None
    counts: dict[str, int] = Field(default_factory=lambda: {"watchlists": 0, "rings": 0, "reports": 0})
    watchlists_preview: list[WatchlistItemDisplay] = Field(default_factory=list)
    rings_preview: list[RingSummary] = Field(default_factory=list)
    reports_preview: list[ReportSummary] = Field(default_factory=list)


def _serialize_watchlist_item(row: dict) -> WatchlistItemDisplay:
    return WatchlistItemDisplay(
        id=str(row.get("id", "")),
        category=row.get("category", "other"),
        type=row.get("type", ""),
        display_label=row.get("display_label"),
        display_value=row.get("display_value"),
        explanation=row.get("explanation"),
        priority=int(row.get("priority", 5)),
        score=float(row["score"]) if row.get("score") is not None else None,
        source_agent=row.get("source_agent"),
        evidence_signal_ids=[str(x) for x in (row.get("evidence_signal_ids") or [])],
    )


def _watchlist_summary_from_items(items: list[dict]) -> WatchlistSummary:
    by_category: dict[str, int] = {}
    for it in items:
        c = it.get("category", "other")
        by_category[c] = by_category.get(c, 0) + 1
    return WatchlistSummary(
        total=len(items),
        by_category=by_category,
        items=[_serialize_watchlist_item(it) for it in items[:100]],
    )


def _watchlist_summary_from_legacy(legacy: list) -> WatchlistSummary:
    """Build summary from legacy watchlists table (watch_type, pattern, reason)."""
    by_category: dict[str, int] = {"other": 0}
    items: list[WatchlistItemDisplay] = []
    for w in legacy[:100]:
        watch_type = getattr(w, "watch_type", None) or "entity_pattern"
        pattern = getattr(w, "pattern", None) or {}
        reason = getattr(w, "reason", None) or ""
        if isinstance(pattern, dict):
            if pattern.get("entity_type") in ("phone", "email", "person"):
                by_category["contact"] = by_category.get("contact", 0) + 1
            elif pattern.get("keywords") or pattern.get("topic_hash"):
                by_category["topic"] = by_category.get("topic", 0) + 1
            else:
                by_category["other"] = by_category.get("other", 0) + 1
        else:
            by_category["other"] = by_category.get("other", 0) + 1
        display_value_raw = ""
        if isinstance(pattern, dict):
            kw = pattern.get("keywords")
            if isinstance(kw, list):
                display_value_raw = ", ".join(str(x) for x in kw)
            elif kw is not None:
                display_value_raw = str(kw)
            else:
                display_value_raw = pattern.get("canonical") or str(pattern.get("canonical_hash", "")) or ""
        display_value_str = (display_value_raw[:500] if display_value_raw else None)
        items.append(WatchlistItemDisplay(
            id=str(getattr(w, "id", "")),
            category="contact" if (isinstance(pattern, dict) and pattern.get("entity_type") in ("phone", "email", "person")) else "topic" if (isinstance(pattern, dict) and (pattern.get("keywords") or pattern.get("topic_hash"))) else "other",
            type=watch_type,
            display_label="Contact" if (isinstance(pattern, dict) and pattern.get("entity_type") in ("phone", "email", "person")) else "Topic" if (isinstance(pattern, dict) and (pattern.get("keywords") or pattern.get("topic_hash"))) else "Item",
            display_value=display_value_str,
            explanation=reason[:2000] if reason else None,
            priority=int(getattr(w, "priority", 5)),
            score=None,
            source_agent=None,
            evidence_signal_ids=[],
        ))
    return WatchlistSummary(total=len(legacy), by_category=by_category, items=items)


@router.get("/summary", response_model=ProtectionSummary)
def get_protection_summary(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Counts + short previews for Protection dashboard cards. Use for nav/summary widgets."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    items = list_active_watchlist_items(supabase, hh_id, limit=10)
    watchlist_summary = _watchlist_summary_from_items(items) if items else WatchlistSummary(total=0, by_category={}, items=[])
    if not items:
        legacy = list_watchlists(hh_id, supabase)
        watchlist_summary = _watchlist_summary_from_legacy(legacy.watchlists)
    try:
        r = supabase.table("rings").select("id, household_id, created_at, updated_at, score, meta, summary_label, summary_text").eq("household_id", hh_id).eq("status", "active").order("updated_at", desc=True).limit(5).execute()
    except Exception:
        r = supabase.table("rings").select("id, household_id, created_at, updated_at, score, meta").eq("household_id", hh_id).order("updated_at", desc=True).limit(5).execute()
    rings_data = r.data or []
    ring_previews = []
    for row in rings_data:
        meta = row.get("meta") or {}
        mc = supabase.table("ring_members").select("id", count="exact").eq("ring_id", row["id"]).execute()
        count = getattr(mc, "count", None) or len(mc.data or [])
        ring_previews.append(RingSummary(id=str(row["id"]), household_id=str(row["household_id"]), created_at=row["created_at"], updated_at=row["updated_at"], score=float(row.get("score", 0)), summary_label=row.get("summary_label") or meta.get("summary_label"), summary_text=row.get("summary_text") or meta.get("summary_text"), members_count=count, meta=meta))
    agents_for_reports = ["evidence_narrative", "continual_calibration", "synthetic_redteam", "model_health"]
    reports_r = supabase.table("agent_runs").select("id, agent_name, started_at, summary_json").in_("agent_name", agents_for_reports).eq("household_id", hh_id).order("started_at", desc=True).limit(20).execute()
    runs = reports_r.data or []
    by_agent: dict[str, dict] = {}
    for run in runs:
        name = run.get("agent_name")
        if name and name not in by_agent:
            by_agent[name] = run
    kind_labels = {"evidence_narrative": "Narrative", "continual_calibration": "Calibration", "synthetic_redteam": "Redteam", "model_health": "Model health"}
    reports_preview = [ReportSummary(kind=kind_labels.get(n, n), last_run_at=by_agent[n].get("started_at") if n in by_agent else None, last_run_id=str(by_agent[n]["id"]) if n in by_agent and by_agent[n].get("id") else None, summary=str((by_agent[n].get("summary_json") or {}).get("summary", ""))[:500] if n in by_agent else None, status=by_agent[n].get("summary_json", {}).get("status") if n in by_agent and isinstance(by_agent[n].get("summary_json"), dict) else None) for n in agents_for_reports]
    last_updated = None
    for row in rings_data:
        u = row.get("updated_at")
        if u and (last_updated is None or u > last_updated):
            last_updated = u
    for it in (items or []):
        u = it.get("updated_at")
        if u and (last_updated is None or u > last_updated):
            last_updated = u
    return ProtectionSummary(
        updated_at=last_updated,
        counts={"watchlists": watchlist_summary.total, "rings": len(rings_data), "reports": len(by_agent)},
        watchlists_preview=watchlist_summary.items[:5],
        rings_preview=ring_previews,
        reports_preview=reports_preview,
    )


@router.get("/overview", response_model=ProtectionOverview)
def get_protection_overview(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Single payload for Protection page: watchlist summary, rings, reports, last_updated."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")

    # Watchlists: prefer watchlist_items, fallback to legacy
    items = list_active_watchlist_items(supabase, hh_id, limit=200)
    if items:
        watchlist_summary = _watchlist_summary_from_items(items)
    else:
        legacy = list_watchlists(hh_id, supabase)
        watchlist_summary = _watchlist_summary_from_legacy(legacy.watchlists)

    # Rings: only active when status column exists; prefer row summary_label/summary_text
    rings_data: list = []
    try:
        rings_r = (
            supabase.table("rings")
            .select("id, household_id, created_at, updated_at, score, meta, summary_label, summary_text")
            .eq("household_id", hh_id)
            .eq("status", "active")
            .order("updated_at", desc=True)
            .limit(50)
            .execute()
        )
        rings_data = rings_r.data or []
    except Exception:
        rings_r = supabase.table("rings").select("id, household_id, created_at, updated_at, score, meta").eq("household_id", hh_id).order("updated_at", desc=True).limit(50).execute()
        rings_data = rings_r.data or []
    ring_summaries = []
    for r in rings_data:
        meta = r.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        mc = supabase.table("ring_members").select("id", count="exact").eq("ring_id", r["id"]).execute()
        members_count = getattr(mc, "count", None) or len(mc.data or [])
        ring_summaries.append(RingSummary(
            id=str(r["id"]),
            household_id=str(r["household_id"]),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            score=float(r.get("score", 0)),
            summary_label=r.get("summary_label") or meta.get("summary_label"),
            summary_text=r.get("summary_text") or meta.get("summary_text"),
            members_count=members_count,
            meta=meta,
        ))

    # Reports: last run per agent type
    agents_for_reports = ["evidence_narrative", "continual_calibration", "synthetic_redteam", "model_health"]
    reports_r = supabase.table("agent_runs").select("id, agent_name, started_at, summary_json").in_("agent_name", agents_for_reports).eq("household_id", hh_id).order("started_at", desc=True).limit(100).execute()
    runs = reports_r.data or []
    by_agent: dict[str, dict] = {}
    for r in runs:
        name = r.get("agent_name")
        if name and name not in by_agent:
            by_agent[name] = r
    kind_labels = {"evidence_narrative": "Narrative", "continual_calibration": "Calibration", "synthetic_redteam": "Redteam", "model_health": "Model health"}
    reports_summary = [
        ReportSummary(
            kind=kind_labels.get(name, name),
            last_run_at=by_agent[name].get("started_at") if name in by_agent else None,
            last_run_id=str(by_agent[name]["id"]) if name in by_agent and by_agent[name].get("id") else None,
            summary=str((by_agent[name].get("summary_json") or {}).get("summary", ""))[:500] if name in by_agent else None,
            status=by_agent[name].get("summary_json", {}).get("status") if name in by_agent and isinstance(by_agent[name].get("summary_json"), dict) else None,
        )
        for name in agents_for_reports
    ]

    last_updated = None
    if rings_data:
        for r in rings_data:
            u = r.get("updated_at")
            if u and (last_updated is None or u > last_updated):
                last_updated = u
    if items:
        for it in items:
            u = it.get("updated_at")
            if u and (last_updated is None or u > last_updated):
                last_updated = u

    return ProtectionOverview(
        watchlist_summary=watchlist_summary,
        rings_summary=ring_summaries,
        reports_summary=reports_summary,
        last_updated_at=last_updated,
        data_freshness={"last_rings": ring_summaries[0].updated_at if ring_summaries else None, "last_watchlists": last_updated},
    )


@router.get("/watchlists", response_model=WatchlistSummary)
def get_protection_watchlists(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
    category: str | None = None,
    type_: str | None = None,
    source_agent: str | None = None,
    limit: int = 200,
):
    """Paginated active watchlist items; optional filter by category, type, source_agent."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    items = list_active_watchlist_items(supabase, hh_id, category=category, type_=type_, source_agent=source_agent, limit=limit)
    if items:
        return _watchlist_summary_from_items(items)
    legacy = list_watchlists(hh_id, supabase)
    return _watchlist_summary_from_legacy(legacy.watchlists)


@router.get("/rings", response_model=list[RingSummary])
def get_protection_rings(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Active rings (canonical view when status column exists) with summary fields."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    try:
        r = (
            supabase.table("rings")
            .select("id, household_id, created_at, updated_at, score, meta, summary_label, summary_text")
            .eq("household_id", hh_id)
            .eq("status", "active")
            .order("updated_at", desc=True)
            .limit(50)
            .execute()
        )
    except Exception:
        r = supabase.table("rings").select("id, household_id, created_at, updated_at, score, meta").eq("household_id", hh_id).order("updated_at", desc=True).limit(50).execute()
    data = r.data or []
    out = []
    for row in data:
        meta = row.get("meta") or {}
        mc = supabase.table("ring_members").select("id", count="exact").eq("ring_id", row["id"]).execute()
        count = getattr(mc, "count", None) or len(mc.data or [])
        out.append(RingSummary(
            id=str(row["id"]),
            household_id=str(row["household_id"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            score=float(row.get("score", 0)),
            summary_label=row.get("summary_label") or meta.get("summary_label"),
            summary_text=row.get("summary_text") or meta.get("summary_text"),
            members_count=count,
            meta=meta,
        ))
    return out


@router.get("/rings/{ring_id}")
def get_protection_ring(
    ring_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Ring detail with members; members include display_label from entities."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    try:
        r = supabase.table("rings").select("id, household_id, created_at, updated_at, score, meta, summary_label, summary_text").eq("id", str(ring_id)).eq("household_id", hh_id).limit(1).execute()
    except Exception:
        r = supabase.table("rings").select("id, household_id, created_at, updated_at, score, meta").eq("id", str(ring_id)).eq("household_id", hh_id).limit(1).execute()
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="Ring not found")
    row = r.data[0]
    members_r = supabase.table("ring_members").select("entity_id, role, first_seen_at, last_seen_at").eq("ring_id", str(ring_id)).execute()
    members_raw = members_r.data or []
    entity_ids = [str(m.get("entity_id", "")) for m in members_raw if m.get("entity_id")]
    display_map = {}
    try:
        from domain.entities.display import get_entity_display_map
        display_map = get_entity_display_map(supabase, hh_id, entity_ids)
    except Exception:
        pass
    members = [
        {
            **m,
            "display_label": display_map.get(str(m.get("entity_id", "")), str(m.get("entity_id", ""))[:8] + "…"),
        }
        for m in members_raw
    ]
    meta = row.get("meta") or {}
    return {
        "id": str(row["id"]),
        "household_id": str(row["household_id"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "score": row.get("score", 0),
        "meta": meta,
        "members": members,
        "summary_label": row.get("summary_label") or meta.get("summary_label"),
        "summary_text": row.get("summary_text") or meta.get("summary_text"),
    }


@router.get("/reports", response_model=list[ReportSummary])
def get_protection_reports(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Human summaries of narrative, calibration, redteam, model health."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    r = supabase.table("agent_runs").select("id, agent_name, started_at, summary_json").in_("agent_name", ["evidence_narrative", "continual_calibration", "synthetic_redteam", "model_health"]).eq("household_id", hh_id).order("started_at", desc=True).limit(100).execute()
    runs = r.data or []
    by_agent: dict[str, dict] = {}
    for row in runs:
        name = row.get("agent_name")
        if name and name not in by_agent:
            by_agent[name] = row
    return [
        ReportSummary(
            kind=name,
            last_run_at=by_agent[name].get("started_at") if name in by_agent else None,
            last_run_id=str(by_agent[name]["id"]) if name in by_agent and by_agent[name].get("id") else None,
            summary=str((by_agent[name].get("summary_json") or {}).get("summary", ""))[:500] if name in by_agent else None,
            status=by_agent[name].get("summary_json", {}).get("status") if name in by_agent and isinstance(by_agent[name].get("summary_json"), dict) else None,
        )
        for name in ["evidence_narrative", "continual_calibration", "synthetic_redteam", "model_health"]
    ]


@router.get("/reports/latest")
def get_protection_reports_latest(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Latest report metadata per type (narrative, calibration, redteam, model_health) + artifact IDs/URLs for Protection."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    r = supabase.table("agent_runs").select("id, agent_name, started_at, summary_json").in_("agent_name", ["evidence_narrative", "continual_calibration", "synthetic_redteam", "model_health"]).eq("household_id", hh_id).order("started_at", desc=True).limit(100).execute()
    runs = r.data or []
    by_agent: dict[str, dict] = {}
    for row in runs:
        name = row.get("agent_name")
        if name and name not in by_agent:
            by_agent[name] = row
    kind_labels = {"evidence_narrative": "narrative", "continual_calibration": "calibration", "synthetic_redteam": "redteam", "model_health": "model_health"}
    out = {}
    for name in ["evidence_narrative", "continual_calibration", "synthetic_redteam", "model_health"]:
        rec = by_agent.get(name)
        out[kind_labels.get(name, name)] = {
            "last_run_at": rec.get("started_at") if rec else None,
            "last_run_id": str(rec["id"]) if rec and rec.get("id") else None,
            "summary": str((rec.get("summary_json") or {}).get("summary", ""))[:500] if rec else None,
            "status": rec.get("summary_json", {}).get("status") if rec and isinstance(rec.get("summary_json"), dict) else None,
        }
    return {"updated_at": max((r.get("started_at") or "") for r in runs) or None, "reports": out}
