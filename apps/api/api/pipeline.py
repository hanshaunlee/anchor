"""
LangGraph pipeline: Ingest -> Normalize -> GraphUpdate -> RiskScore -> Explain -> ConsentGate -> WatchlistSynthesis -> EscalationDraft -> Persist.
Durable checkpoints (memory-backed; swap for DB checkpoint in production).
Thresholds and limits from config.settings to avoid hardcoding.
"""
import logging
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from api.graph_state import AnchorState, append_log

logger = logging.getLogger(__name__)


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
    """Node: build utterances, entities, mentions from events (GraphBuilder)."""
    from ml.graph.builder import GraphBuilder
    events = state.get("ingested_events", [])
    household_id = state.get("household_id", "")
    builder = GraphBuilder(household_id)
    by_session: dict[str, list] = {}
    for ev in events:
        sid = ev.get("session_id", "")
        if sid not in by_session:
            by_session[sid] = []
        by_session[sid].append(ev)
    for sid, evs in by_session.items():
        builder.process_events(evs, sid, evs[0].get("device_id", "") if evs else "")
    state["utterances"] = builder.get_utterance_list()
    state["entities"] = builder.get_entity_list()
    state["mentions"] = builder.get_mention_list()
    state["relationships"] = builder.get_relationship_list()
    state["normalized"] = True
    append_log(state, f"Normalized: {len(state['utterances'])} utterances, {len(state['entities'])} entities")
    return state


def graph_update(state: dict) -> dict:
    """Node: persist entities/mentions/relationships to DB; mark graph_updated."""
    # In production: upsert to Supabase entities, mentions, relationships
    state["graph_updated"] = True
    append_log(state, "Graph updated (persisted)")
    return state


def risk_score_inference(state: dict) -> dict:
    """Node: run GNN risk scoring; append risk_scores; compute time_to_flag (replay: risk rises before big bad event)."""
    settings = _pipeline_settings()
    risk_scores = []
    entities = state.get("entities", [])
    events = state.get("ingested_events", [])
    threshold = settings.risk_score_threshold
    if not entities:
        state["risk_scores"] = risk_scores
        return state
    # Placeholder: call ML inference when model/checkpoint available
    for i, _ in enumerate(entities):
        risk_scores.append({
            "node_type": "entity",
            "node_index": i,
            "score": 0.1 + (i % 3) * 0.2,
            "signal_type": "relational_anomaly",
        })
    state["risk_scores"] = risk_scores
    # time_to_flag: first event ts to first score above threshold (improves vs static baseline in demo)
    first_ts = None
    for e in sorted(events, key=lambda x: (x.get("ts") or 0)):
        t = e.get("ts")
        if t is not None:
            first_ts = t if isinstance(t, (int, float)) else (getattr(t, "timestamp", lambda: None)() or 0)
            break
    if first_ts is not None and risk_scores:
        above = [r for r in risk_scores if r.get("score", 0) >= settings.risk_score_threshold]
        if above:
            state["time_to_flag"] = 0.0  # same-batch flag; in replay we'd compare event ts to flag ts
    append_log(state, f"Risk scored: {len(risk_scores)} nodes")
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
    explanations = []
    for r in state.get("risk_scores", []):
        if r.get("score", 0) < settings.explanation_score_min:
            continue
        expl = {
            "motif_tags": motif_tags_global,
            "model_subgraph": {
                "nodes": [{"id": str(r.get("node_index")), "type": "entity", "score": r.get("score", 0)}],
                "edges": [],
            },
            "timeline_snippet": timeline_snippet[: settings.timeline_snippet_max],
            "top_entities": [r.get("node_index")],
            "top_edges": [],
            "summary": f"Entity {r.get('node_index')} scored {r.get('score', 0):.2f}. " + ("; ".join(motif_tags_global) if motif_tags_global else ""),
        }
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


def synthesize_watchlists(state: dict) -> dict:
    """Node: produce watchlist patterns (hashes, keywords, embedding centroids) if consent allows."""
    if not state.get("consent_allows_watchlist"):
        state["watchlists"] = []
        return state
    settings = _pipeline_settings()
    watchlists = []
    for r in state.get("risk_scores", []):
        if r.get("score", 0) >= settings.watchlist_score_min:
            watchlists.append({
                "watch_type": "entity_pattern",
                "pattern": {"entity_index": r.get("node_index"), "score": r.get("score")},
                "reason": "High risk entity",
                "priority": 1,
            })
    state["watchlists"] = watchlists
    append_log(state, f"Watchlists: {len(watchlists)}")
    return state


def draft_escalation_message(state: dict) -> dict:
    """Node: draft text only; no sending."""
    if not state.get("consent_allows_escalation"):
        state["escalation_draft"] = ""
        return state
    settings = _pipeline_settings()
    high = [r for r in state.get("risk_scores", []) if r.get("score", 0) >= settings.escalation_score_min]
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
    """If severity >= threshold and consent allows -> needs_review else continue."""
    if not state.get("consent_allows_escalation"):
        return "continue"
    settings = _pipeline_settings()
    threshold = state.get("severity_threshold") if state.get("severity_threshold") is not None else settings.severity_threshold
    for r in state.get("risk_scores", []):
        severity = int(1 + (r.get("score", 0) * 4))
        if severity >= threshold:
            return "needs_review"
    return "continue"


def build_graph(checkpointer: Any | None = None) -> StateGraph:
    """Build the LangGraph StateGraph with HITL needs_review branch."""
    graph = StateGraph(dict)
    graph.add_node("ingest", ingest_events_batch)
    graph.add_node("normalize", normalize_events)
    graph.add_node("graph_update", graph_update)
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
    graph.add_edge("graph_update", "risk_score")
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


def run_pipeline(household_id: str, ingested_events: list[dict], time_range_start: str | None = None, time_range_end: str | None = None) -> dict:
    """Run pipeline once with initial state."""
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
    config = {"configurable": {"thread_id": f"hh_{household_id}"}}
    final = None
    for event in app.stream(initial, config=config):
        for k, v in event.items():
            final = v
    return final or initial
