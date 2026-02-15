"""
LangGraph pipeline: Ingest -> Normalize -> GraphUpdate -> RiskScore -> Explain -> ConsentGate -> WatchlistSynthesis -> EscalationDraft -> Persist.
Durable checkpoints (memory-backed; swap for DB checkpoint in production).
Thresholds and limits from config.settings to avoid hardcoding.
PGExplainer is single-source in domain.explainers.pg_service (used only inside risk_scoring_service).
"""
import logging
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from api.graph_state import AnchorState, append_log
from domain.utils.time_utils import event_ts_to_float
from domain.rule_scoring import compute_rule_score

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
    When model is unavailable, uses real rule-only fallback via domain.rule_scoring (no fake placeholders)."""
    from domain.risk_scoring_service import score_risk

    settings = _pipeline_settings()
    risk_scores: list[dict] = []
    entities = state.get("entities", [])
    events = state.get("ingested_events", [])
    if not entities:
        state["risk_scores"] = risk_scores
        state["_model_available"] = False
        return state

    # Motifs (semantic + structural) for fusion and rule-only fallback
    pattern_tags: list[str] = []
    structural_motifs: list[dict] = []
    try:
        from ml.explainers.motifs import extract_motifs
        entity_id_to_canonical = {e["id"]: e.get("canonical", "") for e in entities}
        pattern_tags, timeline_snippet, structural_motifs = extract_motifs(
            state.get("utterances", []),
            state.get("mentions", []),
            entities,
            state.get("relationships", []),
            events,
            entity_id_to_canonical,
        )
        state["_timeline_snippet"] = timeline_snippet
    except Exception:
        structural_motifs = []
    state["_pattern_tags"] = pattern_tags
    state["_structural_motifs"] = structural_motifs

    calibration_params = state.get("calibration_params")
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
        calibration_params=calibration_params,
        pattern_tags=pattern_tags,
        structural_motifs=structural_motifs,
    )

    state["_model_available"] = response.model_available
    state["_risk_scoring_fallback_used"] = response.fallback_used
    if response.model_meta:
        state["_risk_scoring_model_meta"] = response.model_meta.model_dump()
        state["_conformal_q_hat"] = response.model_meta.conformal_q_hat
    if response.model_available and response.scores:
        for item in response.scores:
            risk_scores.append(item.model_dump())
    else:
        state["_risk_scoring_fallback_used"] = "rule_only"
        for i, e in enumerate(entities):
            entity_meta = {"bridges_independent_sets": e.get("bridges_independent_sets", False)}
            if e.get("independence_violation_ratio") is not None:
                entity_meta["independence_violation_ratio"] = e.get("independence_violation_ratio")
            rule_s = compute_rule_score(pattern_tags, structural_motifs, entity_meta)
            risk_scores.append({
                "node_type": "entity",
                "node_index": i,
                "score": round(rule_s, 4),
                "signal_type": "relational_anomaly",
                "model_available": False,
                "rule_score": round(rule_s, 4),
                "decision_rule_used": "rule_only",
                "risk_band": "low_confidence",
            })
    state["risk_scores"] = risk_scores
    append_log(state, f"Risk scored: {len(risk_scores)} nodes (model_available={response.model_available}" + (f", fallback_used={state.get('_risk_scoring_fallback_used')}" if state.get("_risk_scoring_fallback_used") else "") + ")")

    sorted_events = sorted(events or [], key=event_ts_to_float)
    first_ts = event_ts_to_float(sorted_events[0]) if sorted_events else None
    last_ts = event_ts_to_float(sorted_events[-1]) if sorted_events else None
    above = [r for r in risk_scores if r.get("score", 0) >= settings.risk_score_threshold] if risk_scores else []
    if first_ts is not None and last_ts is not None and above:
        state["time_to_flag"] = last_ts - first_ts
    return state


def generate_explanations(state: dict) -> dict:
    """Node: Layer A semantic_pattern_tags + structural_motifs; Layer B model subgraph. explanation_json has semantic_pattern_tags, structural_motifs, model_subgraph, timeline_snippet."""
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
    semantic_pattern_tags = state.get("_pattern_tags", [])
    timeline_snippet = state.get("_timeline_snippet", [])
    structural_motifs = state.get("_structural_motifs", [])
    if extract_motifs and not semantic_pattern_tags and not structural_motifs:
        try:
            semantic_pattern_tags, timeline_snippet, structural_motifs = extract_motifs(
                utterances, mentions, entities, relationships, events, entity_id_to_canonical,
            )
        except Exception:
            pass
    settings = _pipeline_settings()
    model_available = state.get("_model_available", False)
    explanations = []
    for r in state.get("risk_scores", []):
        if r.get("score", 0) < settings.explanation_score_min:
            continue
        expl = {
            "motif_tags": semantic_pattern_tags,
            "semantic_pattern_tags": semantic_pattern_tags,
            "structural_motifs": structural_motifs,
            "model_available": model_available and r.get("model_available", False),
            "timeline_snippet": timeline_snippet[: settings.timeline_snippet_max],
            "top_entities": [r.get("node_index")],
            "top_edges": [],
            "summary": f"Entity {r.get('node_index')} scored {r.get('score', 0):.2f}. " + ("; ".join(semantic_pattern_tags) if semantic_pattern_tags else ""),
        }
        if model_available and r.get("model_available") and r.get("model_subgraph"):
            expl["model_subgraph"] = r["model_subgraph"]
            if r.get("model_evidence_quality"):
                expl["model_evidence_quality"] = r["model_evidence_quality"]
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


def _score_for_severity(r: dict) -> float:
    """Use calibrated_p when available, else fusion_score, else score (so severity uses calibrated probabilities)."""
    return r.get("calibrated_p") if r.get("calibrated_p") is not None else r.get("fusion_score") if r.get("fusion_score") is not None else r.get("score", 0)


def _should_flag_for_escalation(r: dict, settings: Any, effective_threshold: int, conformal_q_hat: float | None) -> bool:
    """Flag when severity >= threshold; when conformal active, flag if 1 - calibrated_p >= q_hat (coverage guarantee)."""
    if conformal_q_hat is not None and r.get("calibrated_p") is not None:
        if 1.0 - r["calibrated_p"] >= conformal_q_hat:
            return int(1 + _score_for_severity(r) * 4) >= effective_threshold
        return False
    s = _score_for_severity(r)
    return s >= settings.escalation_score_min and int(1 + s * 4) >= effective_threshold


def _uncertainty_high(r: dict) -> bool:
    """True when uncertainty is high (e.g. >= 0.2). Used for clarification vs escalate."""
    return (r.get("uncertainty") or 0.2) >= 0.2


def _escalate_vs_clarification(
    r: dict, settings: Any, effective_threshold: int, conformal_q_hat: float | None
) -> tuple[bool, bool]:
    """Returns (should_escalate, needs_clarification). Uncertainty-aware: high p + low unc -> escalate; high p + high unc -> clarification."""
    if not _should_flag_for_escalation(r, settings, effective_threshold, conformal_q_hat):
        return False, False
    s = _score_for_severity(r)
    if int(1 + s * 4) < effective_threshold:
        return False, False
    if _uncertainty_high(r):
        return False, True  # high score but high uncertainty -> clarification question
    return True, False  # high score, low uncertainty -> escalate


def draft_escalation_message(state: dict) -> dict:
    """Node: draft text only; no sending. Uses calibrated_p; conformal when active. Uncertainty-aware: high p + low unc -> escalate; high p + high unc -> clarification."""
    if not state.get("consent_allows_escalation"):
        state["escalation_draft"] = ""
        state["escalation_needs_clarification"] = False
        return state
    settings = _pipeline_settings()
    base = state.get("severity_threshold") or settings.severity_threshold
    adjust = state.get("severity_threshold_adjust") or 0
    effective_threshold = base + adjust
    conformal_q_hat = state.get("_conformal_q_hat")
    to_escalate = []
    to_clarify = []
    for r in state.get("risk_scores", []):
        esc, cl = _escalate_vs_clarification(r, settings, effective_threshold, conformal_q_hat)
        if esc:
            to_escalate.append(r)
        elif cl:
            to_clarify.append(r)
    if to_escalate:
        state["escalation_draft"] = f"Draft escalation: {len(to_escalate)} high-risk signals for review."
    else:
        state["escalation_draft"] = ""
    if to_clarify:
        state["escalation_needs_clarification"] = True
        state["escalation_clarification_count"] = len(to_clarify)
    else:
        state["escalation_needs_clarification"] = False
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
    """If severity >= (base + calibration) and consent allows -> needs_review. Only escalate (low-uncertainty) signals trigger review; high-uncertainty ones need clarification."""
    if not state.get("consent_allows_escalation"):
        return "continue"
    settings = _pipeline_settings()
    base = state.get("severity_threshold") if state.get("severity_threshold") is not None else settings.severity_threshold
    adjust = state.get("severity_threshold_adjust") or 0
    effective_threshold = base + adjust
    conformal_q_hat = state.get("_conformal_q_hat")
    for r in state.get("risk_scores", []):
        esc, _ = _escalate_vs_clarification(r, settings, effective_threshold, conformal_q_hat)
        if esc:
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
    calibration_params: dict | None = None,
) -> dict:
    """Run pipeline once with initial state. severity_threshold_adjust and calibration_params from household_calibration (worker passes them)."""
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
    if calibration_params is not None:
        initial["calibration_params"] = calibration_params
    config = {"configurable": {"thread_id": f"hh_{household_id}"}}
    final = None
    for event in app.stream(initial, config=config):
        for k, v in event.items():
            final = v
    return final or initial
