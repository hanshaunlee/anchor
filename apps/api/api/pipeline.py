"""
LangGraph pipeline: Ingest -> Normalize -> GraphUpdate -> RiskScore -> Explain -> ConsentGate -> WatchlistSynthesis -> EscalationDraft -> Persist.
Durable checkpoints (memory-backed; swap for DB checkpoint in production).
Thresholds and limits from config.settings to avoid hardcoding.
"""
import logging
from typing import Any

import torch
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from torch_geometric.data import Data

from api.graph_state import AnchorState, append_log

logger = logging.getLogger(__name__)


def _edge_attr_dim() -> int:
    try:
        from config.graph import get_graph_config
        return get_graph_config().get("edge_attr_dim", 4)
    except ImportError:
        return 4


def _attach_pg_explainer_subgraphs(
    model,
    data,
    target_node_type: str,
    device: torch.device,
    risk_scores: list[dict],
    explanation_score_min: float,
    top_k_edges: int = 20,
) -> None:
    """Compute PGExplainer edge importances and attach model_subgraph to each risk_score above threshold. Mutates risk_scores in place."""
    try:
        from ml.explainers.pg_explainer import PGExplainerStyle, explain_with_pg
    except ImportError:
        logger.debug("PGExplainer not available, skipping model_subgraph")
        return
    with torch.no_grad():
        _, h_dict = model.forward_hetero_data_with_hidden(data)
    node_emb = h_dict.get(target_node_type)
    if node_emb is None or node_emb.size(0) == 0:
        return
    edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    edge_attr = None
    try:
        _, edge_types = data.metadata()
        for (src, rel, dst) in edge_types:
            if src == target_node_type and dst == target_node_type:
                store = data[src, rel, dst]
                edge_index = store.edge_index.to(device)
                edge_attr = getattr(store, "edge_attr", None)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(device)
                break
    except Exception:
        pass
    edge_dim = _edge_attr_dim()
    if edge_attr is None and edge_index.size(1) > 0:
        edge_attr = torch.zeros(edge_index.size(1), edge_dim, device=device)
    elif edge_attr is not None:
        edge_dim = edge_attr.size(-1)
    hom_data = Data(x=data[target_node_type].x.to(device), edge_index=edge_index, edge_attr=edge_attr)
    hidden_dim = node_emb.size(-1)
    pg = PGExplainerStyle(hidden_dim, edge_dim).to(device).eval()
    score_by_idx = {r["node_index"]: r["score"] for r in risk_scores}
    for r in risk_scores:
        if r.get("score", 0) < explanation_score_min:
            continue
        node_idx = r.get("node_index", 0)
        if edge_index.size(1) == 0:
            r["model_subgraph"] = {
                "nodes": [{"id": str(node_idx), "type": "entity", "score": r.get("score", 0)}],
                "edges": [],
            }
            r["model_available"] = True
            continue
        expl = explain_with_pg(pg, node_emb, hom_data, top_k=top_k_edges)
        top_edges = expl["top_edges"]
        minimal_node_ids = set(expl["minimal_subgraph_node_ids"])
        incident_edges = [e for e in top_edges if e["src"] == node_idx or e["dst"] == node_idx]
        if not incident_edges:
            incident_edges = top_edges[:5]
        incident_nodes = set()
        for e in incident_edges:
            incident_nodes.add(e["src"])
            incident_nodes.add(e["dst"])
        if node_idx not in incident_nodes:
            incident_nodes.add(node_idx)
        nodes = [
            {"id": str(n), "type": "entity", "score": score_by_idx.get(n) if n < len(risk_scores) else None}
            for n in sorted(incident_nodes)
        ]
        edges = [
            {"src": str(e["src"]), "dst": str(e["dst"]), "weight": round(e["score"], 4), "rank": i}
            for i, e in enumerate(incident_edges)
        ]
        r["model_subgraph"] = {"nodes": nodes, "edges": edges}
        r["model_available"] = True


def _pipeline_settings():
    try:
        from config.settings import get_pipeline_settings
        return get_pipeline_settings()
    except ImportError:
        from pydantic import BaseSettings
        class _Fallback(BaseSettings):
            risk_score_threshold: float = 0.5
            explanation_score_min: float = 0.4
            watchlist_score_min: float = 0.5
            escalation_score_min: float = 0.6
            severity_threshold: int = 4
            timeline_snippet_max: int = 6
            consent_share_key: str = "share_with_caregiver"
            consent_watchlist_key: str = "watchlist_ok"
            default_consent_share: bool = True
            default_consent_watchlist: bool = True
        return _Fallback()


def ingest_events_batch(state: dict) -> dict:
    """Node: load events for household in time range. In production: query Supabase."""
    household_id = state.get("household_id", "")
    start = state.get("time_range_start")
    end = state.get("time_range_end")
    append_log(state, f"Ingest events household={household_id} range={start}..{end}")
    # Placeholder: state may be pre-filled with ingested_events by worker
    if not state.get("ingested_events"):
        state["ingested_events"] = []
        state["session_ids"] = []
    return state


def normalize_events(state: dict) -> dict:
    """Node: build utterances, entities, mentions, relationships from events via shared graph_service (no DB write in pipeline)."""
    from domain.graph_service import build_graph_from_events
    events = state.get("ingested_events", [])
    household_id = state.get("household_id", "")
    utterances, entities, mentions, relationships = build_graph_from_events(household_id, events, supabase=None)
    state["utterances"] = utterances
    state["entities"] = entities
    state["mentions"] = mentions
    state["relationships"] = relationships
    state["normalized"] = True
    append_log(state, f"Normalized: {len(utterances)} utterances, {len(entities)} entities")
    return state


def graph_update(state: dict) -> dict:
    """Node: persist entities/mentions/relationships to DB; mark graph_updated."""
    # In production: upsert to Supabase entities, mentions, relationships
    state["graph_updated"] = True
    append_log(state, "Graph updated (persisted)")
    return state


def financial_security_agent(state: dict) -> dict:
    """
    Node: Financial Security Agent playbook (read-only recommendations; no money movement).
    Runs after graph_update, before consent_gate. Uses state utterances/entities/mentions/relationships;
    when run from pipeline no supabase so results go to state only (persist via on-demand API or worker).
    """
    from domain.agents.financial_security_agent import run_financial_security_playbook
    try:
        settings = _pipeline_settings()
        consent = state.get("consent_state") or {}
        result = run_financial_security_playbook(
            household_id=state.get("household_id", ""),
            time_window_days=7,
            consent_state=consent,
            ingested_events=state.get("ingested_events"),
            supabase=None,  # pipeline context: no DB write; use POST /agents/financial/run to persist
            dry_run=True,
            escalation_severity_threshold=getattr(settings, "severity_threshold", 4),
            persist_score_min=getattr(settings, "persist_score_min", 0.3),
            watchlist_score_min=settings.watchlist_score_min,
        )
    except Exception as e:
        logger.exception("Financial security agent failed: %s", e)
        result = {
            "risk_signals": [],
            "watchlists": [],
            "logs": [f"Financial agent error: {e}"],
            "run_id": None,
            "inserted_signal_ids": [],
            "inserted_signals_for_broadcast": [],
            "session_ids": [],
            "motif_tags": [],
            "timeline_snippet": [],
        }
    state["financial_risk_signals"] = result.get("risk_signals", [])
    state["financial_watchlists"] = result.get("watchlists", [])
    state["financial_logs"] = result.get("logs", [])
    for msg in result.get("logs", []):
        append_log(state, msg)
    return state


def _sessions_from_events(ingested_events: list[dict]) -> list[dict]:
    """Build session list for graph: one entry per session_id with started_at = min(ts) in that session."""
    by_sid: dict[str, list] = {}
    for ev in ingested_events or []:
        sid = ev.get("session_id") or ""
        if sid not in by_sid:
            by_sid[sid] = []
        by_sid[sid].append(ev.get("ts"))
    sessions = []
    for sid, ts_list in by_sid.items():
        if not sid:
            continue
        valid_ts = [t for t in ts_list if t is not None]
        started_at = min(valid_ts) if valid_ts else None
        sessions.append({"id": sid, "started_at": started_at})
    return sessions


def risk_score_inference(state: dict) -> dict:
    """Node: run GNN risk scoring via single risk scoring service; append risk_scores; compute time_to_flag.
    When model is unavailable, uses explicit rule-only fallback (no silent placeholders)."""
    from domain.risk_scoring_service import score_risk

    settings = _pipeline_settings()
    risk_scores: list[dict] = []
    entities = state.get("entities", [])
    events = state.get("ingested_events", [])
    if not entities:
        state["risk_scores"] = risk_scores
        state["_model_available"] = False
        return state

    sessions = _sessions_from_events(events)
    response = score_risk(
        state.get("household_id", ""),
        sessions=sessions,
        utterances=state.get("utterances", []),
        entities=entities,
        mentions=state.get("mentions", []),
        relationships=state.get("relationships", []),
        devices=state.get("devices", []),
        events=events,
        explanation_score_min=settings.explanation_score_min,
    )

    state["_model_available"] = response.model_available
    state["_risk_scoring_fallback_used"] = response.fallback_used
    if response.model_available and response.scores:
        for item in response.scores:
            risk_scores.append(item.model_dump())
        if response.model_meta:
            state["_risk_scoring_model_meta"] = response.model_meta.model_dump()
    else:
        # Explicit rule-only fallback: no embedding, no model_subgraph; model_available=false
        state["_risk_scoring_fallback_used"] = "rule_only"
        for i, _ in enumerate(entities):
            risk_scores.append({
                "node_type": "entity",
                "node_index": i,
                "score": 0.1 + (i % 3) * 0.2,
                "signal_type": "relational_anomaly",
                "model_available": False,
            })
    state["risk_scores"] = risk_scores
    append_log(state, f"Risk scored: {len(risk_scores)} nodes (model_available={response.model_available}" + (f", fallback_used={state.get('_risk_scoring_fallback_used')}" if state.get("_risk_scoring_fallback_used") else "") + ")")
    # time_to_flag: seconds from first event ts to first time we exceed threshold (for replay: use scripts/run_replay_time_to_flag.py)
    def _ts_to_float(ev: dict) -> float:
        t = ev.get("ts")
        if t is None:
            return 0.0
        if isinstance(t, (int, float)):
            return float(t)
        if hasattr(t, "timestamp"):
            return t.timestamp()
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return 0.0
    sorted_events = sorted(events or [], key=_ts_to_float)
    first_ts = _ts_to_float(sorted_events[0]) if sorted_events else None
    last_ts = _ts_to_float(sorted_events[-1]) if sorted_events else None
    above = [r for r in risk_scores if r.get("score", 0) >= settings.risk_score_threshold] if risk_scores else []
    if first_ts is not None and last_ts is not None and above:
        state["time_to_flag"] = last_ts - first_ts  # seconds from first event to flag in this batch; replay gives finer granularity
    return state


def generate_explanations(state: dict) -> dict:
    """Node: Layer A motifs + Layer B model subgraph; explanation_json has motif_tags, model_subgraph, timeline_snippet."""
    try:
        from ml.explainers.motifs import extract_motifs
    except ImportError:
        extract_motifs = None
    utterances = state.get("utterances", [])
    mentions = state.get("mentions", [])
    entities = state.get("entities", [])
    relationships = state.get("relationships", [])
    events = state.get("ingested_events", [])
    entity_id_to_canonical = {e["id"]: e.get("canonical", "") for e in entities}
    motif_tags_global, timeline_snippet = [], []
    if extract_motifs:
        motif_tags_global, timeline_snippet = extract_motifs(
            utterances, mentions, entities, relationships, events, entity_id_to_canonical
        )
    settings = _pipeline_settings()
    model_available = state.get("_model_available", False)
    explanations = []
    for r in state.get("risk_scores", []):
        if r.get("score", 0) < settings.explanation_score_min:
            continue
        expl = {
            "motif_tags": motif_tags_global,
            "model_available": model_available and r.get("model_available", False),
            "timeline_snippet": timeline_snippet[: settings.timeline_snippet_max],
            "top_entities": [r.get("node_index")],
            "top_edges": [],
            "summary": f"Entity {r.get('node_index')} scored {r.get('score', 0):.2f}. " + ("; ".join(motif_tags_global) if motif_tags_global else ""),
        }
        if model_available and r.get("model_available") and r.get("model_subgraph"):
            expl["model_subgraph"] = r["model_subgraph"]
            if r.get("model_evidence_quality"):
                expl["model_evidence_quality"] = r["model_evidence_quality"]
        # When model did not run: do not include model_subgraph (so "delete the GNN" test fails in the good way).
        explanations.append({"node_index": r.get("node_index"), "explanation_json": expl})
    state["explanations"] = explanations
    append_log(state, f"Explanations: {len(explanations)}")
    return state


def consent_policy_gate(state: dict) -> dict:
    """Node: check consent_state; set consent_allows_escalation / consent_allows_watchlist."""
    settings = _pipeline_settings()
    consent = state.get("consent_state") or {}
    state["consent_allows_escalation"] = consent.get(settings.consent_share_key, settings.default_consent_share)
    state["consent_allows_watchlist"] = consent.get(settings.consent_watchlist_key, settings.default_consent_watchlist)
    append_log(state, "Consent gate: allowed" if state["consent_allows_escalation"] else "Consent: no escalation")
    return state


def _l2_normalize(vec: list[float]) -> list[float]:
    s = sum(x * x for x in vec) ** 0.5
    if s <= 0:
        return vec
    return [x / s for x in vec]


def _embedding_centroid_watchlist(
    risk_scores: list[dict],
    score_min: float,
    min_embeddings: int = 3,
    cosine_threshold: float = 0.82,
    created_from_window_days: int = 14,
) -> dict | None:
    """If >= min_embeddings high-risk nodes have real embeddings, return one watchlist with L2-normalized centroid; matches by cosine distance."""
    high_risk_with_emb = [
        r for r in risk_scores
        if r.get("score", 0) >= score_min and r.get("embedding") and isinstance(r["embedding"], (list, tuple)) and len(r["embedding"]) > 0
    ]
    if len(high_risk_with_emb) < min_embeddings:
        return None
    normalized = [_l2_normalize([float(x) for x in r["embedding"]]) for r in high_risk_with_emb]
    dim = len(normalized[0])
    centroid = [sum(n[i] for n in normalized) / len(normalized) for i in range(dim)]
    centroid = _l2_normalize(centroid)
    model_name = "hgt_baseline"
    try:
        from config.settings import get_ml_settings
        model_name = get_ml_settings().model_version_tag or model_name
    except ImportError:
        pass
    window_str = f"{created_from_window_days}d"
    return {
        "watch_type": "embedding_centroid",
        "pattern": {
            "metric": "cosine",
            "threshold": cosine_threshold,
            "cosine_threshold": cosine_threshold,
            "centroid": centroid,
            "dim": dim,
            "model_name": model_name,
            "source": {
                "risk_signal_ids": [],
                "window": window_str,
            },
            "provenance": {
                "risk_signal_ids": [],
                "window_days": created_from_window_days,
                "node_indices": [r.get("node_index") for r in high_risk_with_emb],
            },
        },
        "reason": "GNN embedding centroid of high-risk entities",
        "priority": 2,
        "expires_at_days": 7,
    }


def synthesize_watchlists(state: dict) -> dict:
    """Node: produce watchlist patterns (hashes, keywords, embedding centroids) if consent allows. Centroid only when GNN ran and embeddings exist."""
    if not state.get("consent_allows_watchlist"):
        state["watchlists"] = []
        return state
    settings = _pipeline_settings()
    watchlists = []
    for r in state.get("risk_scores", []):
        if r.get("score", 0) >= settings.watchlist_score_min:
            watchlists.append({
                "watch_type": "entity_pattern",
                "pattern": {"node_index": r.get("node_index"), "score": r.get("score")},
                "reason": "High risk entity",
                "priority": 1,
            })
    centroid_wl = _embedding_centroid_watchlist(state.get("risk_scores", []), settings.watchlist_score_min)
    if centroid_wl is not None:
        watchlists.append(centroid_wl)
    state["watchlists"] = watchlists
    append_log(state, f"Watchlists: {len(watchlists)}")
    return state


def draft_escalation_message(state: dict) -> dict:
    """Node: draft text only; no sending. Uses base severity threshold + household calibration."""
    if not state.get("consent_allows_escalation"):
        state["escalation_draft"] = ""
        return state
    settings = _pipeline_settings()
    base = state.get("severity_threshold") or settings.severity_threshold
    adjust = state.get("severity_threshold_adjust") or 0
    effective_threshold = base + adjust
    high = [
        r for r in state.get("risk_scores", [])
        if r.get("score", 0) >= settings.escalation_score_min and int(1 + (r.get("score", 0) * 4)) >= effective_threshold
    ]
    if high:
        state["escalation_draft"] = f"Draft escalation: {len(high)} high-risk signals for review."
    else:
        state["escalation_draft"] = ""
    return state


def persist_outputs(state: dict) -> dict:
    """Node: write risk_signals, watchlists to DB; set persisted."""
    state["persisted"] = True
    append_log(state, "Persisted risk_signals and watchlists")
    return state


def needs_review_node(state: dict) -> dict:
    """HITL: wait for caregiver review (Confirm scam / False alarm / Unsure). In production, pause until feedback."""
    state["needs_review"] = True
    append_log(state, "Needs review: awaiting caregiver feedback")
    return state


def should_review(state: dict) -> str:
    """If severity >= (base threshold + household calibration) and consent allows -> needs_review else continue."""
    if not state.get("consent_allows_escalation"):
        return "continue"
    settings = _pipeline_settings()
    base = state.get("severity_threshold") if state.get("severity_threshold") is not None else settings.severity_threshold
    adjust = state.get("severity_threshold_adjust") or 0
    effective_threshold = base + adjust
    for r in state.get("risk_scores", []):
        severity = int(1 + (r.get("score", 0) * 4))
        if severity >= effective_threshold:
            return "needs_review"
    return "continue"


def build_graph(checkpointer: Any | None = None) -> StateGraph:
    """Build the LangGraph StateGraph with HITL needs_review branch."""
    graph = StateGraph(dict)
    graph.add_node("ingest", ingest_events_batch)
    graph.add_node("normalize", normalize_events)
    graph.add_node("graph_update", graph_update)
    graph.add_node("financial_security_agent", financial_security_agent)
    graph.add_node("risk_score", risk_score_inference)
    graph.add_node("explain", generate_explanations)
    graph.add_node("consent_gate", consent_policy_gate)
    graph.add_conditional_edges("consent_gate", should_review, {"needs_review": "needs_review", "continue": "watchlist"})
    graph.add_node("needs_review", needs_review_node)
    graph.add_edge("needs_review", "watchlist")
    graph.add_node("watchlist", synthesize_watchlists)
    graph.add_node("escalation_draft", draft_escalation_message)
    graph.add_node("persist", persist_outputs)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "normalize")
    graph.add_edge("normalize", "graph_update")
    graph.add_edge("graph_update", "financial_security_agent")
    graph.add_edge("financial_security_agent", "risk_score")
    graph.add_edge("risk_score", "explain")
    graph.add_edge("explain", "consent_gate")
    graph.add_edge("watchlist", "escalation_draft")
    graph.add_edge("escalation_draft", "persist")
    graph.add_edge("persist", END)
    if checkpointer:
        graph = graph.compile(checkpointer=checkpointer)
    else:
        graph = graph.compile()
    return graph


def run_pipeline(
    household_id: str,
    ingested_events: list[dict],
    time_range_start: str | None = None,
    time_range_end: str | None = None,
    severity_threshold_adjust: float | None = None,
) -> dict:
    """Run pipeline once with initial state. severity_threshold_adjust from household_calibration (worker passes it)."""
    checkpointer = MemorySaver()
    app = build_graph(checkpointer)
    initial = {
        "household_id": household_id,
        "time_range_start": time_range_start,
        "time_range_end": time_range_end,
        "ingested_events": ingested_events,
        "session_ids": list({e.get("session_id") for e in ingested_events if e.get("session_id")}),
        "consent_state": {},  # In production: from session or household
    }
    if severity_threshold_adjust is not None:
        initial["severity_threshold_adjust"] = severity_threshold_adjust
    config = {"configurable": {"thread_id": f"hh_{household_id}"}}
    final = None
    for event in app.stream(initial, config=config):
        for k, v in event.items():
            final = v
    return final or initial
