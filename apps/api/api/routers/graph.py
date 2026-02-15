"""
Graph evidence and optional Neo4j sync. Serves the household evidence subgraph
for the UI "Graph view" and triggers Neo4j mirror when requested.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import RiskSignalDetailSubgraph, SubgraphEdge, SubgraphNode

router = APIRouter(prefix="/graph", tags=["graph"])


def _household_id(supabase: Client, user_id: str) -> str | None:
    u = supabase.table("users").select("household_id").eq("id", user_id).single().execute()
    return u.data["household_id"] if u.data else None


def _fetch_events_for_household(supabase: Client, household_id: str, limit: int = 2000) -> list[dict]:
    """Fetch events for household from Supabase (recent sessions first). Used to build evidence graph."""
    sess = (
        supabase.table("sessions")
        .select("id")
        .eq("household_id", household_id)
        .order("started_at", desc=True)
        .limit(200)
        .execute()
    )
    session_ids = [s["id"] for s in (sess.data or [])]
    if not session_ids:
        return []
    ev = (
        supabase.table("events")
        .select("id, session_id, device_id, ts, seq, event_type, payload")
        .in_("session_id", session_ids)
        .order("ts")
        .limit(limit)
        .execute()
    )
    return list(ev.data or [])


def _fetch_persisted_graph(supabase: Client, household_id: str) -> tuple[list[dict], list[dict]]:
    """Load persisted entities and relationships from Supabase for this household. Returns (entities, relationships) in builder-like shape."""
    ent = (
        supabase.table("entities")
        .select("id, entity_type, canonical")
        .eq("household_id", household_id)
        .execute()
    )
    entities = [
        {"id": str(e["id"]), "entity_type": e.get("entity_type", "entity"), "canonical": e.get("canonical") or ""}
        for e in (ent.data or [])
    ]
    if not entities:
        return [], []
    rel = (
        supabase.table("relationships")
        .select("src_entity_id, dst_entity_id, rel_type, weight")
        .eq("household_id", household_id)
        .execute()
    )
    id_set = {e["id"] for e in entities}
    relationships = [
        {
            "src_entity_id": str(r["src_entity_id"]),
            "dst_entity_id": str(r["dst_entity_id"]),
            "rel_type": r.get("rel_type", "CO_OCCURS"),
            "weight": float(r.get("weight", 1.0)),
            "count": 1,
        }
        for r in (rel.data or [])
        if str(r.get("src_entity_id", "")) in id_set and str(r.get("dst_entity_id", "")) in id_set
    ]
    return entities, relationships


def _build_graph_from_events(household_id: str, events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Build graph from events via shared graph_service (no persist for evidence endpoint)."""
    from domain.graph_service import build_graph_from_events
    _, entities, _, relationships = build_graph_from_events(household_id, events, supabase=None)
    return entities, relationships


def _merge_evidence_graph(
    event_entities: list[dict],
    event_relationships: list[dict],
    db_entities: list[dict],
    db_relationships: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Merge event-built graph with persisted entities/relationships from Supabase."""
    event_ids = {str(e.get("id", "")) for e in event_entities}
    entities = event_entities + [e for e in db_entities if e.get("id") not in event_ids]
    all_ids = {str(e.get("id", "")) for e in entities}
    relationships = event_relationships + [
        r for r in db_relationships
        if r.get("src_entity_id") in all_ids and r.get("dst_entity_id") in all_ids
    ]
    return entities, relationships


def _to_subgraph(entities: list[dict], relationships: list[dict]) -> RiskSignalDetailSubgraph:
    """Convert builder entities/relationships to UI subgraph schema."""
    nodes: list[SubgraphNode] = []
    seen: set[str] = set()
    for e in entities:
        eid = str(e.get("id", ""))
        if eid in seen:
            continue
        seen.add(eid)
        nodes.append(SubgraphNode(
            id=eid,
            type=e.get("entity_type", "entity"),
            label=e.get("canonical"),
            score=None,
        ))
    edges: list[SubgraphEdge] = []
    for r in relationships:
        src = str(r.get("src_entity_id", ""))
        dst = str(r.get("dst_entity_id", ""))
        if src not in seen or dst not in seen:
            continue
        edges.append(SubgraphEdge(
            src=src,
            dst=dst,
            type=r.get("rel_type", "CO_OCCURS"),
            weight=float(r.get("weight", 1.0)),
            rank=int(r.get("count", 0)),
        ))
    return RiskSignalDetailSubgraph(nodes=nodes, edges=edges)


@router.get("/evidence", response_model=RiskSignalDetailSubgraph)
def get_evidence_graph(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Return the household evidence subgraph for the Graph view. Built from Supabase: events (recent sessions) plus persisted entities/relationships."""
    hh_id = _household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    events = _fetch_events_for_household(supabase, hh_id)
    event_entities, event_relationships = _build_graph_from_events(hh_id, events)
    db_entities, db_relationships = _fetch_persisted_graph(supabase, hh_id)
    entities, relationships = _merge_evidence_graph(
        event_entities, event_relationships, db_entities, db_relationships
    )
    return _to_subgraph(entities, relationships)


@router.post("/sync-neo4j")
def sync_neo4j(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
) -> dict[str, Any]:
    """Mirror current household evidence subgraph to Neo4j. No-op if Neo4j not configured."""
    from api.neo4j_sync import neo4j_enabled, sync_evidence_graph_to_neo4j
    if not neo4j_enabled():
        return {"ok": False, "message": "Neo4j not configured (set NEO4J_URI)"}
    hh_id = _household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    events = _fetch_events_for_household(supabase, hh_id)
    event_entities, event_relationships = _build_graph_from_events(hh_id, events)
    db_entities, db_relationships = _fetch_persisted_graph(supabase, hh_id)
    entities, relationships = _merge_evidence_graph(
        event_entities, event_relationships, db_entities, db_relationships
    )
    synced = sync_evidence_graph_to_neo4j(hh_id, entities, relationships)
    return {"ok": synced, "message": "Synced to Neo4j" if synced else "Sync failed", "entities": len(entities), "relationships": len(relationships)}


@router.get("/neo4j-status")
def neo4j_status() -> dict[str, Any]:
    """Return whether Neo4j is configured, Browser URL, and optional connect URL with auth for pre-filling login."""
    from api.neo4j_sync import neo4j_enabled
    from api.config import settings
    from urllib.parse import quote_plus
    enabled = neo4j_enabled()
    out: dict[str, Any] = {"enabled": enabled}
    if enabled:
        uri = (getattr(settings, "neo4j_uri", None) or "").strip()
        if "localhost" in uri or "127.0.0.1" in uri:
            out["browser_url"] = "http://localhost:7474"
            user = (getattr(settings, "neo4j_user", None) or "neo4j").strip()
            password = (getattr(settings, "neo4j_password", None) or "").strip()
            if user and password:
                safe_user = quote_plus(user)
                safe_pass = quote_plus(password)
                out["connect_url"] = f"neo4j://{safe_user}:{safe_pass}@localhost:7687"
                out["password"] = password  # Shown in UI for copy-paste (Neo4j Browser cannot pre-fill password)
        else:
            out["browser_url"] = None
    return out
