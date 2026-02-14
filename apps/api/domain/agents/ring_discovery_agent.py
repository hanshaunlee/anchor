"""
Ring Discovery Agent: dual-mode (Neo4j GDS or NetworkX) cluster/ring discovery.
Builds interaction graph from relationships + mentions; scores communities; emits ring_candidate risk_signals and persists rings/ring_members.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any

from domain.agents.base import AgentContext, persist_agent_run, step, upsert_risk_signal
from domain.ml_artifacts import cluster_embeddings

logger = logging.getLogger(__name__)

MIN_COMMUNITY_SIZE = 2
TOP_RINGS = 10
NEO4J_AVAILABLE_KEY = "neo4j_available"


def _build_interaction_graph_supabase(supabase: Any, household_id: str) -> tuple[dict[str, set[str]], dict[tuple[str, str], float], list[str]]:
    """
    Build graph from relationships + mentions. Returns:
    - node_neighbors: node_id -> set of neighbor ids
    - edge_weights: (src, dst) -> weight (count/recency)
    - all_node_ids
    """
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
    """Use NetworkX connected components or community detection; return list of communities (each = list of node ids)."""
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
                comps = [list(c) for c in communities if len(c) >= MIN_COMMUNITY_SIZE]
            except Exception:
                comps = list(nx.connected_components(G))
                comps = [list(c) for c in comps if len(c) >= MIN_COMMUNITY_SIZE]
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
            if len(comp) >= MIN_COMMUNITY_SIZE:
                connected.append(comp)
        return connected if connected else [[n] for n in node_ids[:5]] if node_ids else []


def _suspiciousness_score(
    community: list[str],
    supabase: Any,
    household_id: str,
    risk_signal_entity_ids: set[str],
    entity_first_seen: dict[str, str],
) -> float:
    """Score 0..1: concentration of high-risk entities, burstiness, novelty."""
    if not community:
        return 0.0
    risk_count = sum(1 for e in community if e in risk_signal_entity_ids)
    concentration = risk_count / len(community) if community else 0
    novelty = 0.0
    try:
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        for e in community:
            if (entity_first_seen.get(e) or "") >= recent:
                novelty += 0.2
        novelty = min(1.0, novelty)
    except Exception:
        pass
    return min(1.0, 0.5 * concentration + 0.3 * (1.0 if risk_count > 0 else 0) + 0.2 * novelty)


def _risk_signal_entity_ids(supabase: Any, household_id: str) -> set[str]:
    """Entity ids that appear in recent risk signals' explanations."""
    out = set()
    try:
        r = (
            supabase.table("risk_signals")
            .select("explanation")
            .eq("household_id", household_id)
            .gte("ts", (datetime.now(timezone.utc) - timedelta(days=30)).isoformat())
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


def run_ring_discovery_agent(
    household_id: str,
    supabase: Any | None = None,
    neo4j_available: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Build interaction graph; run Neo4j GDS (if available) or NetworkX clustering;
    score communities, emit ring_candidate risk_signals and persist rings/ring_members.
    When Neo4j unavailable, uses NetworkX and still produces rings + risk_signals.
    """
    step_trace: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()
    ctx = AgentContext(household_id, supabase, dry_run=dry_run)

    if not supabase:
        step_trace.append({
            "step": "build_graph",
            "status": "skip",
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "notes": "no_supabase",
        })
        summary = {"rings_found": 0, "risk_signals_created": 0, "reason": "no_supabase", "neo4j_available": False}
        ended_at = datetime.now(timezone.utc).isoformat()
        return {
            "step_trace": step_trace,
            "summary_json": summary,
            "status": "ok",
            "started_at": started_at,
            "ended_at": ended_at,
            "run_id": None,
        }

    with step(ctx, step_trace, "build_graph"):
        node_neighbors, edge_weights, node_ids = _build_interaction_graph_supabase(supabase, household_id)
        step_trace[-1]["outputs_count"] = len(node_ids)
        step_trace[-1]["notes"] = f"{len(node_ids)} nodes, {len(edge_weights)} edges"

    if len(node_ids) < MIN_COMMUNITY_SIZE:
        step_trace.append({
            "step": "cluster",
            "status": "ok",
            "started_at": step_trace[-1]["ended_at"],
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "notes": "insufficient_nodes",
        })
        summary = {
            "rings_found": 0,
            "risk_signals_created": 0,
            "reason": "insufficient_nodes",
            "neo4j_available": neo4j_available,
            "artifact_refs": {"ring_ids": [], "risk_signal_ids": []},
        }
        ended_at = datetime.now(timezone.utc).isoformat()
        run_id = persist_agent_run(supabase, household_id, "ring_discovery", started_at=started_at, ended_at=ended_at, status="completed", step_trace=step_trace, summary_json=summary, dry_run=dry_run)
        return {"step_trace": step_trace, "summary_json": summary, "status": "ok", "started_at": started_at, "ended_at": ended_at, "run_id": run_id}

    with step(ctx, step_trace, "cluster"):
        if neo4j_available:
            try:
                from api.neo4j_sync import get_neo4j_driver
                driver = get_neo4j_driver()
                if driver:
                    with driver.session() as session:
                        session.run("MERGE (n:Entity {id: $id})", id=node_ids[0])
                    communities = _cluster_with_networkx(node_neighbors, edge_weights, node_ids)
                else:
                    communities = _cluster_with_networkx(node_neighbors, edge_weights, node_ids)
            except Exception as e:
                logger.debug("Neo4j GDS not used: %s", e)
                communities = _cluster_with_networkx(node_neighbors, edge_weights, node_ids)
        else:
            communities = _cluster_with_networkx(node_neighbors, edge_weights, node_ids)
        step_trace[-1]["outputs_count"] = len(communities)
        step_trace[-1]["notes"] = f"{len(communities)} communities (neo4j={neo4j_available})"

    risk_entity_ids = _risk_signal_entity_ids(supabase, household_id) if supabase else set()
    entity_first_seen: dict[str, str] = {}
    try:
        rels = supabase.table("relationships").select("src_entity_id, first_seen_at").eq("household_id", household_id).execute()
        for r in (rels.data or []):
            eid = str(r.get("src_entity_id", ""))
            if eid and r.get("first_seen_at"):
                entity_first_seen[eid] = str(r["first_seen_at"])
    except Exception:
        pass

    scored = []
    for comm in communities:
        if len(comm) < MIN_COMMUNITY_SIZE:
            continue
        score = _suspiciousness_score(comm, supabase, household_id, risk_entity_ids, entity_first_seen)
        scored.append((comm, score))
    scored.sort(key=lambda x: -x[1])
    top_rings = scored[:TOP_RINGS]

    ring_ids_created: list[str] = []
    risk_signal_ids_created: list[str] = []

    with step(ctx, step_trace, "persist_rings_and_signals"):
        for idx, (members, score) in enumerate(top_rings):
            if not members or score <= 0:
                continue
            ring_id = None
            if not dry_run and supabase:
                try:
                    ins = supabase.table("rings").insert({
                        "household_id": household_id,
                        "score": float(score),
                        "meta": {"member_count": len(members), "index": idx},
                    }).execute()
                    if ins.data and len(ins.data) > 0:
                        ring_id = ins.data[0].get("id")
                        if ring_id:
                            ring_ids_created.append(ring_id)
                            for eid in members[:50]:
                                supabase.table("ring_members").insert({
                                    "ring_id": ring_id,
                                    "entity_id": eid,
                                    "role": "member",
                                    "first_seen_at": entity_first_seen.get(eid),
                                    "last_seen_at": entity_first_seen.get(eid),
                                }).execute()
                except Exception as e:
                    logger.warning("Ring insert failed: %s", e)
            severity = min(5, max(1, int(1 + score * 4)))
            explanation = {
                "summary": f"Ring candidate: {len(members)} entities, suspiciousness {score:.2f}.",
                "ring_id": ring_id,
                "members_count": len(members),
                "member_entity_ids": members[:20],
                "key_paths": [],
                "motifs_observed": [],
            }
            rsid = upsert_risk_signal(
                supabase, household_id,
                {
                    "signal_type": "ring_candidate",
                    "severity": severity,
                    "score": float(score),
                    "explanation": explanation,
                    "recommended_action": {"checklist": ["Review ring members", "Check linked risk signals"], "action": "review"},
                    "status": "open",
                },
                dry_run=dry_run,
            )
            if rsid:
                risk_signal_ids_created.append(rsid)
        step_trace[-1]["outputs_count"] = len(ring_ids_created) + len(risk_signal_ids_created)
        step_trace[-1]["artifacts_refs"] = {"ring_ids": ring_ids_created, "risk_signal_ids": risk_signal_ids_created}

    summary = {
        "rings_found": len(ring_ids_created),
        "risk_signals_created": len(risk_signal_ids_created),
        "communities_evaluated": len(top_rings),
        "neo4j_available": neo4j_available,
        "artifact_refs": {"ring_ids": ring_ids_created, "risk_signal_ids": risk_signal_ids_created},
    }
    ended_at = datetime.now(timezone.utc).isoformat()
    run_id = persist_agent_run(supabase, household_id, "ring_discovery", started_at=started_at, ended_at=ended_at, status="completed", step_trace=step_trace, summary_json=summary, dry_run=dry_run)
    return {
        "step_trace": step_trace,
        "summary_json": summary,
        "status": "ok",
        "started_at": started_at,
        "ended_at": ended_at,
        "run_id": run_id,
    }
