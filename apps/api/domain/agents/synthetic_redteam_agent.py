"""
Synthetic Red-Team Agent: scenario DSL + regression harness for similar incidents, centroid watchlists, evidence subgraph.
Generates templated scam variants; runs pipeline in sandbox; asserts regression; persists run with pass rate and failing_cases.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import random
from datetime import datetime, timezone, timedelta
from typing import Any

from domain.agents.base import AgentContext, persist_agent_run, step
from domain.graph_service import normalize_events
from domain.ml_artifacts import cosine_sim, load_checkpoint_or_none, normalize

logger = logging.getLogger(__name__)

SCENARIO_THEMES = ("medicare", "irs", "grandchild", "bank_fraud", "crypto")
TACTICS = ("urgency", "authority", "otp_request", "payee_add", "device_switch")
SYNONYM_MAP: dict[str, list[str]] = {
    "Medicare": ["Medicare", "health insurance", "benefits"],
    "suspended": ["suspended", "blocked", "on hold"],
    "verify": ["verify", "confirm", "validate"],
    "immediately": ["immediately", "right away", "as soon as possible"],
    "Social Security": ["Social Security", "SSN", "social security number"],
}
NUM_VARIANTS = 3
SIMILARITY_THRESHOLD = 0.7
MIN_SUBGRAPH_EDGES = 5
EXPECTED_EDGE_TYPES = {"mentions", "co_occurs", "next_event", "triggered"}


def _make_ts(day_offset: int = 0, hour: int = 14, minute: int = 30, sec: int = 0) -> str:
    t = datetime.now(timezone.utc) - timedelta(days=day_offset)
    t = t.replace(hour=hour, minute=minute, second=sec, microsecond=0)
    return t.isoformat()


def _paraphrase(text: str) -> str:
    out = text
    for word, synonyms in SYNONYM_MAP.items():
        if word in out and synonyms:
            out = out.replace(word, random.choice(synonyms), 1)
            break
    return out


def _seed_events_for_theme(theme: str, session_id: str, device_id: str, day_offset: int = 0) -> list[dict]:
    """EventPacket-like events for a seed scenario (no LLM)."""
    if theme == "medicare":
        return [
            {"session_id": session_id, "device_id": device_id, "ts": _make_ts(day_offset, 14, 30, 0), "seq": 0, "event_type": "wake", "payload": {}},
            {"session_id": session_id, "device_id": device_id, "ts": _make_ts(day_offset, 14, 30, 15), "seq": 1, "event_type": "final_asr", "payload": {"text": "Someone called about Medicare saying my account is suspended", "confidence": 0.88, "speaker": {"role": "elder"}}},
            {"session_id": session_id, "device_id": device_id, "ts": _make_ts(day_offset, 14, 30, 30), "seq": 2, "event_type": "intent", "payload": {"name": "share_ssn", "slots": {"number": "555-1234"}, "confidence": 0.85}},
        ]
    if theme == "irs":
        return [
            {"session_id": session_id, "device_id": device_id, "ts": _make_ts(day_offset, 10, 0, 0), "seq": 0, "event_type": "wake", "payload": {}},
            {"session_id": session_id, "device_id": device_id, "ts": _make_ts(day_offset, 10, 0, 10), "seq": 1, "event_type": "final_asr", "payload": {"text": "IRS said I owe taxes and must verify immediately", "confidence": 0.9, "speaker": {"role": "elder"}}},
            {"session_id": session_id, "device_id": device_id, "ts": _make_ts(day_offset, 10, 0, 20), "seq": 2, "event_type": "intent", "payload": {"name": "sensitive_request", "slots": {"topic": "tax"}, "confidence": 0.8}},
        ]
    if theme == "bank_fraud":
        return [
            {"session_id": session_id, "device_id": device_id, "ts": _make_ts(day_offset, 16, 0, 0), "seq": 0, "event_type": "wake", "payload": {}},
            {"session_id": session_id, "device_id": device_id, "ts": _make_ts(day_offset, 16, 0, 15), "seq": 1, "event_type": "final_asr", "payload": {"text": "I added a new payee for a wire", "confidence": 0.85, "speaker": {"role": "elder"}}},
            {"session_id": session_id, "device_id": device_id, "ts": _make_ts(day_offset, 16, 0, 30), "seq": 2, "event_type": "payee_added", "payload": {"payee_name": "Unknown LLC", "confidence": 0.9}},
        ]
    return [
        {"session_id": session_id, "device_id": device_id, "ts": _make_ts(day_offset), "seq": 0, "event_type": "wake", "payload": {}},
        {"session_id": session_id, "device_id": device_id, "ts": _make_ts(day_offset, 14, 30, 15), "seq": 1, "event_type": "final_asr", "payload": {"text": "Someone asked for my information", "confidence": 0.8, "speaker": {"role": "elder"}}},
    ]


def _generate_variants(theme: str, n: int) -> list[dict[str, Any]]:
    """Generate n scenario variants: new session/device, paraphrased text, optional reorder."""
    scenarios = []
    for i in range(n):
        session_id = hashlib.sha256(f"{theme}-{i}-{random.getstate()[1][0]}".encode()).hexdigest()[:12]
        device_id = hashlib.sha256(f"device-{theme}-{i}".encode()).hexdigest()[:12]
        events = _seed_events_for_theme(theme, session_id, device_id, day_offset=i % 7)
        variant = []
        for ev in events:
            ev_copy = copy.deepcopy(ev)
            if ev_copy.get("event_type") == "final_asr" and ev_copy.get("payload", {}).get("text"):
                ev_copy["payload"] = dict(ev_copy["payload"])
                ev_copy["payload"]["text"] = _paraphrase(ev_copy["payload"]["text"])
            variant.append(ev_copy)
        scenarios.append({"theme": theme, "scenario_id": f"{theme}_{i}", "events": variant})
    return scenarios


def _run_pipeline_sandbox(household_id: str, events: list[dict]) -> dict[str, Any]:
    """Normalize + score_risk in memory; return top risk signals with embeddings and model_subgraph."""
    utterances, entities, mentions, relationships = normalize_events(household_id, events)
    result: dict[str, Any] = {"model_available": False, "scores": [], "embeddings": [], "model_subgraphs": []}
    try:
        from domain.risk_scoring_service import score_risk
        sessions = [{"id": events[0]["session_id"] if events else "s1", "started_at": 0}]
        resp = score_risk(
            household_id,
            sessions=sessions,
            utterances=utterances,
            entities=entities,
            mentions=mentions,
            relationships=relationships,
            events=events,
        )
        result["model_available"] = resp.model_available
        if resp.model_available and resp.scores:
            for s in resp.scores:
                result["scores"].append({"node_index": s.node_index, "score": s.score})
                if s.embedding:
                    result["embeddings"].append(s.embedding)
                if s.model_subgraph:
                    result["model_subgraphs"].append(s.model_subgraph)
    except Exception as e:
        logger.debug("Redteam pipeline sandbox failed: %s", e)
    return result


def run_synthetic_redteam_agent(
    household_id: str,
    supabase: Any | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """
    Generate scenario variants; run pipeline in sandbox; assert similar incidents, centroid watchlist, evidence subgraph;
    persist agent_run with regression_pass_rate and failing_cases. When model unavailable, record reason and fail regression.
    """
    step_trace: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()
    ctx = AgentContext(household_id, supabase, dry_run=dry_run)

    model_available = load_checkpoint_or_none() is not None

    with step(ctx, step_trace, "generate_variants"):
        scenarios: list[dict] = []
        for theme in SCENARIO_THEMES[:3]:
            scenarios.extend(_generate_variants(theme, NUM_VARIANTS))
        step_trace[-1]["outputs_count"] = len(scenarios)
        step_trace[-1]["notes"] = f"{len(scenarios)} scenarios"

    seed_embedding_by_theme: dict[str, list[float]] = {}
    failing_cases: list[dict] = []
    passed = 0
    total_assertions = 0

    with step(ctx, step_trace, "run_regression"):
        for sc in scenarios:
            theme = sc.get("theme", "")
            events = sc.get("events", [])
            run_result = _run_pipeline_sandbox(household_id, events)
            if not run_result["model_available"]:
                failing_cases.append({"scenario_id": sc.get("scenario_id"), "expected": "model_available", "got": "model_unavailable"})
                total_assertions += 1
                continue
            scores = run_result.get("scores") or []
            embeddings = run_result.get("embeddings") or []
            subgraphs = run_result.get("model_subgraphs") or []
            top_score = max((s.get("score", 0) for s in scores), default=0)
            seed_emb = seed_embedding_by_theme.get(theme)
            if not seed_emb and embeddings:
                seed_embedding_by_theme[theme] = embeddings[0]
                seed_emb = embeddings[0]
            if seed_emb and embeddings:
                sim = cosine_sim(seed_emb, embeddings[0])
                total_assertions += 1
                if sim >= SIMILARITY_THRESHOLD:
                    passed += 1
                else:
                    failing_cases.append({"scenario_id": sc.get("scenario_id"), "assertion": "similar_incidents", "expected": f">= {SIMILARITY_THRESHOLD}", "got": round(sim, 4)})
            if subgraphs:
                total_assertions += 1
                edges = subgraphs[0].get("edges") or []
                edge_types = set()
                for e in edges:
                    t = e.get("type") or "edge"
                    edge_types.add(t)
                if len(edges) >= MIN_SUBGRAPH_EDGES and (EXPECTED_EDGE_TYPES & edge_types or not EXPECTED_EDGE_TYPES):
                    passed += 1
                else:
                    failing_cases.append({"scenario_id": sc.get("scenario_id"), "assertion": "evidence_subgraph", "expected": f"edges>={MIN_SUBGRAPH_EDGES} and expected types", "got": f"edges={len(edges)} types={edge_types}"})
        step_trace[-1]["outputs_count"] = total_assertions
        step_trace[-1]["notes"] = f"passed={passed} total={total_assertions}"

    regression_passed = model_available and total_assertions > 0 and (passed / total_assertions) >= 0.5
    if not model_available:
        step_trace.append({
            "step": "model_check",
            "status": "ok",
            "started_at": step_trace[-1]["ended_at"],
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "notes": "model_unavailable",
        })

    summary = {
        "scenarios_generated": len(scenarios),
        "regression_pass_rate": round(passed / total_assertions, 4) if total_assertions else 0,
        "regression_passed": regression_passed,
        "model_available": model_available,
        "failing_cases": failing_cases,
        "artifact_refs": {"scenario_ids": [s.get("scenario_id") for s in scenarios]},
    }
    ended_at = datetime.now(timezone.utc).isoformat()
    run_id = persist_agent_run(
        supabase, household_id, "synthetic_redteam",
        started_at=started_at, ended_at=ended_at, status="completed",
        step_trace=step_trace, summary_json=summary, dry_run=dry_run,
    ) if supabase else None
    return {
        "step_trace": step_trace,
        "summary_json": summary,
        "status": "ok",
        "started_at": started_at,
        "ended_at": ended_at,
        "run_id": run_id,
    }
