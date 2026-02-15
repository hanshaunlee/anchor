"""
Ring Discovery Agent: Ring + Connector + Escalation.
Ten steps: data acquisition, build interaction graph, candidate ring discovery,
ring scoring, connector/bridge analysis, evidence subgraph per ring, output artifacts,
watchlists derived from ring, optional escalation draft, persist & UI hooks.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any

from domain.agents.base import (
    AgentContext,
    persist_agent_run,
    persist_agent_run_ctx,
    step,
    upsert_risk_signal,
    upsert_risk_signal_ctx,
    upsert_watchlists,
)

logger = logging.getLogger(__name__)

NEO4J_AVAILABLE_KEY = "neo4j_available"


def _agent_settings():
    try:
        from config.settings import get_agent_settings
        return get_agent_settings()
    except Exception:
        class _F:
            ring_min_community_size = 2
            ring_top_rings = 10
            ring_novelty_days = 7
            ring_lookback_days = 30
        return _F()


def _build_interaction_graph_supabase(supabase: Any, household_id: str) -> tuple[dict[str, set[str]], dict[tuple[str, str], float], list[str]]:
    node_neighbors: dict[str, set[str]] = defaultdict(set)
    edge_weights: dict[tuple[str, str], float] = defaultdict(float)
    nodes: set[str] = set()
    try:
        rels = (
            supabase.table("relationships")
            .select("src_entity_id, dst_entity_id, rel_type, weight, last_seen_at")
            .eq("household_id", household_id)
            .execute()
        )
        for r in (rels.data or []):
            src = str(r.get("src_entity_id", ""))
            dst = str(r.get("dst_entity_id", ""))
            if not src or not dst:
                continue
            nodes.add(src)
            nodes.add(dst)
            w = float(r.get("weight") or 1.0)
            edge_weights[(src, dst)] += w
            edge_weights[(dst, src)] += w
            node_neighbors[src].add(dst)
            node_neighbors[dst].add(src)
    except Exception as e:
        logger.debug("Ring agent: relationships fetch failed %s", e)
    try:
        mentions = (
            supabase.table("mentions")
            .select("session_id, entity_id")
            .execute()
        )
        session_entities: dict[str, set[str]] = defaultdict(set)
        for m in (mentions.data or []):
            eid = str(m.get("entity_id", ""))
            sid = str(m.get("session_id", ""))
            if eid and sid:
                session_entities[sid].add(eid)
                nodes.add(eid)
        for sid, eids in session_entities.items():
            for e1 in eids:
                for e2 in eids:
                    if e1 < e2:
                        edge_weights[(e1, e2)] += 0.5
                        edge_weights[(e2, e1)] += 0.5
                        node_neighbors[e1].add(e2)
                        node_neighbors[e2].add(e1)
    except Exception as e:
        logger.debug("Ring agent: mentions co-occur failed %s", e)
    return dict(node_neighbors), dict(edge_weights), list(nodes)


def _cluster_with_networkx(
    node_neighbors: dict[str, set[str]],
    edge_weights: dict[tuple[str, str], float],
    node_ids: list[str],
) -> list[list[str]]:
    try:
        import networkx as nx
        G = nx.Graph()
        for n in node_ids:
            G.add_node(n)
        for (a, b), w in edge_weights.items():
            if a in node_ids and b in node_ids and a != b:
                G.add_edge(a, b, weight=w)
        if G.number_of_edges() == 0:
            comps = [[n] for n in G.nodes()]
        else:
            try:
                from networkx.algorithms import community
                communities = community.greedy_modularity_communities(G)
                min_size = _agent_settings().ring_min_community_size
                comps = [list(c) for c in communities if len(c) >= min_size]
            except Exception:
                comps = list(nx.connected_components(G))
                min_size = _agent_settings().ring_min_community_size
                comps = [list(c) for c in comps if len(c) >= min_size]
        if not comps:
            comps = [[n] for n in node_ids[:20]] if node_ids else []
        return comps
    except ImportError:
        connected: list[list[str]] = []
        seen = set()
        for n in node_ids:
            if n in seen:
                continue
            comp = []
            stack = [n]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp.append(cur)
                for nb in node_neighbors.get(cur, []):
                    if nb not in seen:
                        stack.append(nb)
            if len(comp) >= _agent_settings().ring_min_community_size:
                connected.append(comp)
        return connected if connected else [[n] for n in node_ids[:5]] if node_ids else []


def _suspiciousness_score(
    community: list[str],
    supabase: Any,
    household_id: str,
    risk_signal_entity_ids: set[str],
    entity_first_seen: dict[str, str],
) -> float:
    if not community:
        return 0.0
    risk_count = sum(1 for e in community if e in risk_signal_entity_ids)
    concentration = risk_count / len(community) if community else 0
    novelty = 0.0
    try:
        recent = (datetime.now(timezone.utc) - timedelta(days=_agent_settings().ring_novelty_days)).isoformat()
        for e in community:
            if (entity_first_seen.get(e) or "") >= recent:
                novelty += 0.2
        novelty = min(1.0, novelty)
    except Exception:
        pass
    return min(1.0, 0.5 * concentration + 0.3 * (1.0 if risk_count > 0 else 0) + 0.2 * novelty)


def _risk_signal_entity_ids(supabase: Any, household_id: str) -> set[str]:
    out = set()
    try:
        r = (
            supabase.table("risk_signals")
            .select("explanation")
            .eq("household_id", household_id)
            .gte("ts", (datetime.now(timezone.utc) - timedelta(days=_agent_settings().ring_lookback_days)).isoformat())
            .limit(200)
            .execute()
        )
        for row in (r.data or []):
            expl = row.get("explanation") or {}
            for sub in ("subgraph", "model_subgraph"):
                sg = expl.get(sub) or {}
                for n in sg.get("nodes") or []:
                    eid = n.get("id")
                    if eid:
                        out.add(str(eid))
    except Exception:
        pass
    return out


def _connector_bridge_analysis(
    community: list[str],
    node_neighbors: dict[str, set[str]],
    edge_weights: dict[tuple[str, str], float],
) -> list[dict]:
    """Top 1-3 connector nodes: betweenness; bridge risk. Returns list of {entity_id, betweenness, bridge_risk}."""
    if len(community) < 2:
        return []
    try:
        import networkx as nx
        G = nx.Graph()
        for n in community:
            G.add_node(n)
        for (a, b), w in edge_weights.items():
            if a in community and b in community and a != b:
                G.add_edge(a, b, weight=w)
        if G.number_of_edges() == 0:
            return []
        betweenness = nx.betweenness_centrality(G, weight="weight")
        articulation = set(nx.articulation_points(G))
        scored = []
        for n in community:
            b = betweenness.get(n, 0.0)
            bridge = 1.0 if n in articulation else 0.0
            bridge_risk = round(b * 0.7 + bridge * 0.3, 4)
            scored.append({"entity_id": n, "betweenness": round(b, 4), "bridge_risk": bridge_risk})
        scored.sort(key=lambda x: -x["betweenness"])
        return scored[:3]
    except Exception as e:
        logger.debug("Connector analysis failed: %s", e)
        return []


def _ring_evidence_subgraph(
    community: list[str],
    edge_weights: dict[tuple[str, str], float],
    top_n_edges: int = 20,
) -> dict[str, Any]:
    """Stable ids: nodes, top edges by weight, list of edge descriptors."""
    comm_set = set(community)
    edges_in = [(a, b, w) for (a, b), w in edge_weights.items() if a in comm_set and b in comm_set and a != b]
    edges_in.sort(key=lambda x: -x[2])
    top_edges = edges_in[:top_n_edges]
    return {
        "nodes": [{"id": n} for n in community],
        "edges": [{"src": a, "dst": b, "weight": w} for a, b, w in top_edges],
        "top_edge_count": len(top_edges),
    }


def run_ring_discovery_playbook(
    ctx: AgentContext,
    *,
    neo4j_available: bool = False,
) -> dict[str, Any]:
    """
    Ten-step Ring + Connector + Escalation agent.
    Returns step_trace, summary_json, status, run_id, artifacts_refs.
    """
    step_trace: list[dict] = []
    started_at = ctx.now.isoformat()
    summary_json: dict[str, Any] = {"headline": "Ring Discovery", "key_metrics": {}, "key_findings": [], "recommended_actions": [], "artifact_refs": {}}
    artifacts_refs: dict[str, Any] = {"ring_ids": [], "risk_signal_ids": [], "watchlist_ids": []}
    run_id: str | None = None

    if not ctx.supabase:
        step_trace.append({"step": "build_graph", "status": "skip", "started_at": started_at, "ended_at": ctx.now.isoformat(), "notes": "no_supabase"})
        summary_json["rings_found"] = 0
        summary_json["risk_signals_created"] = 0
        summary_json["reason"] = "no_supabase"
        summary_json["neo4j_available"] = False
        run_id = persist_agent_run_ctx(ctx, "ring_discovery", "completed", step_trace, summary_json, artifacts_refs)
        return {"step_trace": step_trace, "summary_json": summary_json, "status": "ok", "run_id": run_id, "started_at": started_at, "ended_at": ctx.now.isoformat()}

    # Step 1 — Data acquisition
    with step(ctx, step_trace, "data_acquisition"):
        rels = ctx.supabase.table("relationships").select("src_entity_id, dst_entity_id", count="exact").eq("household_id", ctx.household_id).execute()
        ment = ctx.supabase.table("mentions").select("entity_id", count="exact").limit(1).execute()
        risk_count = ctx.supabase.table("risk_signals").select("id", count="exact").eq("household_id", ctx.household_id).gte("ts", (ctx.now - timedelta(days=_agent_settings().ring_lookback_days)).isoformat()).execute()
        emb = ctx.supabase.table("risk_signal_embeddings").select("risk_signal_id").eq("household_id", ctx.household_id).eq("has_embedding", True).limit(1).execute()
        watchlist_hits = 0
        try:
            w = ctx.supabase.table("watchlists").select("id", count="exact").eq("household_id", ctx.household_id).execute()
            watchlist_hits = getattr(w, "count", None) or len(w.data or [])
        except Exception:
            pass
        step_trace[-1]["outputs_count"] = 4
        step_trace[-1]["notes"] = f"rels, mentions, risk_signals, embeddings_available={bool(emb.data)}"

    # Step 2 — Build interaction graph
    with step(ctx, step_trace, "build_graph"):
        node_neighbors, edge_weights, node_ids = _build_interaction_graph_supabase(ctx.supabase, ctx.household_id)
        risk_entity_ids = _risk_signal_entity_ids(ctx.supabase, ctx.household_id)
        for (a, b), w in list(edge_weights.items()):
            if a in risk_entity_ids or b in risk_entity_ids:
                edge_weights[(a, b)] = w * 1.2
        step_trace[-1]["outputs_count"] = len(node_ids)
        step_trace[-1]["notes"] = f"{len(node_ids)} nodes, {len(edge_weights)} edges"

    if len(node_ids) < _agent_settings().ring_min_community_size:
        step_trace.append({"step": "cluster", "status": "ok", "started_at": step_trace[-1]["ended_at"], "ended_at": ctx.now.isoformat(), "notes": "insufficient_nodes"})
        summary_json["rings_found"] = 0
        summary_json["risk_signals_created"] = 0
        summary_json["reason"] = "insufficient_nodes"
        summary_json["neo4j_available"] = neo4j_available
        summary_json["artifact_refs"] = artifacts_refs
        run_id = persist_agent_run_ctx(ctx, "ring_discovery", "completed", step_trace, summary_json, artifacts_refs)
        return {"step_trace": step_trace, "summary_json": summary_json, "status": "ok", "run_id": run_id, "started_at": started_at, "ended_at": ctx.now.isoformat()}

    # Step 3 — Candidate ring discovery
    with step(ctx, step_trace, "candidate_ring_discovery"):
        if neo4j_available:
            try:
                from api.neo4j_sync import get_neo4j_driver
                driver = get_neo4j_driver()
                if driver:
                    with driver.session() as session:
                        session.run("MERGE (n:Entity {id: $id})", id=node_ids[0])
                communities = _cluster_with_networkx(node_neighbors, edge_weights, node_ids)
            except Exception as e:
                logger.debug("Neo4j GDS not used: %s", e)
                step_trace[-1]["notes"] = f"neo4j_fallback: {e}"
                communities = _cluster_with_networkx(node_neighbors, edge_weights, node_ids)
        else:
            communities = _cluster_with_networkx(node_neighbors, edge_weights, node_ids)
        step_trace[-1]["outputs_count"] = len(communities)
        step_trace[-1]["notes"] = f"{len(communities)} communities (neo4j={neo4j_available})"

    entity_first_seen: dict[str, str] = {}
    try:
        rels = ctx.supabase.table("relationships").select("src_entity_id, first_seen_at").eq("household_id", ctx.household_id).execute()
        for r in (rels.data or []):
            eid = str(r.get("src_entity_id", ""))
            if eid and r.get("first_seen_at"):
                entity_first_seen[eid] = str(r["first_seen_at"])
    except Exception:
        pass
    risk_entity_ids = _risk_signal_entity_ids(ctx.supabase, ctx.household_id)

    # Step 4 — Ring scoring
    with step(ctx, step_trace, "ring_scoring"):
        scored = []
        for comm in communities:
            if len(comm) < _agent_settings().ring_min_community_size:
                continue
            score = _suspiciousness_score(comm, ctx.supabase, ctx.household_id, risk_entity_ids, entity_first_seen)
            scored.append((comm, score))
        scored.sort(key=lambda x: -x[1])
        top_rings = scored[:_agent_settings().ring_top_rings]
        step_trace[-1]["outputs_count"] = len(top_rings)
        step_trace[-1]["notes"] = f"top {len(top_rings)} rings"

    # Step 5 — Connector/bridge analysis
    with step(ctx, step_trace, "connector_bridge_analysis"):
        ring_connectors: list[list[dict]] = []
        for comm, _ in top_rings:
            conns = _connector_bridge_analysis(comm, node_neighbors, edge_weights)
            ring_connectors.append(conns)
        step_trace[-1]["outputs_count"] = len(ring_connectors)

    # Step 6 — Evidence subgraph per ring
    with step(ctx, step_trace, "evidence_subgraph_per_ring"):
        ring_subgraphs = [_ring_evidence_subgraph(comm, edge_weights) for comm, _ in top_rings]
        step_trace[-1]["outputs_count"] = len(ring_subgraphs)

    # Step 7 — Output artifacts (rings, ring_members, risk_signal); dedupe by fingerprint/overlap
    ring_ids_created: list[str] = []
    risk_signal_ids_created: list[str] = []
    with step(ctx, step_trace, "output_artifacts"):
        for idx, ((members, score), connectors, subgraph) in enumerate(zip(top_rings, ring_connectors, ring_subgraphs)):
            if not members or score <= 0:
                continue
            ring_id = None
            if not ctx.dry_run and ctx.supabase:
                try:
                    from domain.rings.service import upsert_ring_by_fingerprint
                    meta = {"member_count": len(members), "index": idx, "top_connectors": connectors, "evidence_edge_count": subgraph.get("top_edge_count", 0)}
                    summary_label = f"Cluster of {len(members)} entities"
                    ring_id = upsert_ring_by_fingerprint(
                        ctx.supabase,
                        ctx.household_id,
                        members,
                        float(score),
                        meta=meta,
                        summary_label=summary_label,
                        summary_text=None,
                        signals_count=0,
                    )
                    if ring_id and ring_id not in ring_ids_created:
                        ring_ids_created.append(ring_id)
                except Exception as e:
                    logger.warning("Ring upsert failed: %s", e)
            severity = min(5, max(1, int(1 + score * 4)))
            # Skip creating a risk signal for very low suspiciousness to avoid noise (e.g. 0.08)
            if severity < 2 and score < 0.15:
                continue
            explanation = {
                "summary": f"Ring candidate: {len(members)} entities, suspiciousness {score:.2f}.",
                "ring_id": ring_id,
                "members_count": len(members),
                "member_entity_ids": members[:20],
                "ring_evidence": subgraph,
                "top_connectors": connectors,
                "key_paths": [],
                "motifs_observed": [],
            }
            rsid = upsert_risk_signal_ctx(ctx, "ring_candidate", severity, float(score), explanation, {"checklist": ["Review ring members", "Open ring view"], "action": "review"}, "open")
            if rsid:
                risk_signal_ids_created.append(rsid)
        artifacts_refs["ring_ids"] = ring_ids_created
        artifacts_refs["risk_signal_ids"] = risk_signal_ids_created
        step_trace[-1]["outputs_count"] = len(ring_ids_created) + len(risk_signal_ids_created)

    # Step 8 — Watchlists derived from ring
    with step(ctx, step_trace, "watchlists_derived"):
        watchlist_items: list[dict] = []
        if ctx.consent_for("share_with_caregiver", True) and top_rings and not ctx.dry_run:
            members_top = top_rings[0][0][:10]
            for eid in members_top[:5]:
                watchlist_items.append({"watch_type": "entity_pattern", "pattern": {"entity_id": eid}, "reason": "Ring discovery: high-risk ring member", "priority": 1})
        if watchlist_items:
            wids = upsert_watchlists(ctx, watchlist_items)
            artifacts_refs["watchlist_ids"] = wids
        step_trace[-1]["outputs_count"] = len(watchlist_items)

    # Step 9 — Optional escalation draft
    with step(ctx, step_trace, "escalation_draft"):
        escalation_draft = None
        if top_rings and top_rings[0][1] >= 0.6 and ctx.consent_for("share_with_caregiver", True):
            comm, score = top_rings[0]
            escalation_draft = {"summary": f"High-scoring ring ({len(comm)} entities, score {score:.2f}). Review ring view and top connector entities.", "top_entity_ids": comm[:5]}
        summary_json["escalation_draft"] = escalation_draft
        step_trace[-1]["outputs_count"] = 1 if escalation_draft else 0

    # Step 10 — Persist & UI hooks
    with step(ctx, step_trace, "persist_ui"):
        summary_json["rings_found"] = len(ring_ids_created)
        summary_json["risk_signals_created"] = len(risk_signal_ids_created)
        summary_json["neo4j_available"] = neo4j_available
        summary_json["headline"] = f"Found {len(ring_ids_created)} ring(s); {len(risk_signal_ids_created)} alert(s) created"
        summary_json["key_metrics"] = {"rings_found": len(ring_ids_created), "risk_signals_created": len(risk_signal_ids_created)}
        summary_json["key_findings"] = [f"Rings: {len(ring_ids_created)}. Open ring view from /graph or /alerts."]
        summary_json["recommended_actions"] = ["View rings on /graph", "Open ring view from alert detail"]
        summary_json["artifact_refs"] = artifacts_refs
        run_id = persist_agent_run_ctx(ctx, "ring_discovery", "completed", step_trace, summary_json, artifacts_refs)
        step_trace[-1]["artifacts_refs"] = artifacts_refs

    return {
        "step_trace": step_trace,
        "summary_json": summary_json,
        "status": "ok",
        "started_at": started_at,
        "ended_at": ctx.now.isoformat(),
        "run_id": run_id,
        "artifacts_refs": artifacts_refs,
        "watchlist_items": watchlist_items,
    }


def run_ring_discovery_agent(
    household_id: str,
    supabase: Any | None = None,
    neo4j_available: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Wrapper: build ctx and call run_ring_discovery_playbook."""
    ctx = AgentContext(household_id, supabase, dry_run=dry_run)
    out = run_ring_discovery_playbook(ctx, neo4j_available=neo4j_available)
    return {
        "step_trace": out["step_trace"],
        "summary_json": out["summary_json"],
        "status": out["status"],
        "started_at": out["started_at"],
        "ended_at": out["ended_at"],
        "run_id": out.get("run_id"),
        "artifacts_refs": out.get("artifacts_refs"),
    }
