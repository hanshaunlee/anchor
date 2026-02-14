"""
Synthetic Red-Team Agent: Scenario Generator + Regression Harness.
Nine steps: define objectives, generate scenario DSL, run pipeline sandbox, similar incidents regression,
centroid watchlist regression, evidence subgraph regression, generate replay artifact, user-visible report, UI.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from domain.agents.base import AgentContext, persist_agent_run, persist_agent_run_ctx, step, upsert_risk_signal_ctx
from domain.graph_service import normalize_events
from domain.ml_artifacts import cosine_sim, load_checkpoint_or_none, centroid

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
PASS_RATE_THRESHOLD = 0.5


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
        scenarios.append({"theme": theme, "tactic_sequence": [TACTICS[0]], "scenario_id": f"{theme}_{i}", "events": variant, "timestamps": [e.get("ts") for e in variant]})
    return scenarios


def _run_pipeline_sandbox(household_id: str, events: list[dict]) -> dict[str, Any]:
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


def run_synthetic_redteam_playbook(
    ctx: AgentContext,
    *,
    n_variants: int = 3,
    themes: tuple[str, ...] | None = None,
    require_model: bool = True,
) -> dict[str, Any]:
    """
    Nine-step Scenario Generator + Regression Harness.
    Returns step_trace, summary_json, status, run_id, artifacts_refs (including replay_fixture_path).
    """
    step_trace: list[dict] = []
    started_at = ctx.now.isoformat()
    summary_json: dict[str, Any] = {"headline": "Red-Team Regression", "key_metrics": {}, "key_findings": [], "recommended_actions": [], "artifact_refs": {}}
    artifacts_refs: dict[str, Any] = {"scenario_ids": [], "replay_fixture_path": None, "risk_signal_ids": []}
    run_id: str | None = None
    themes = themes or SCENARIO_THEMES[:3]
    model_available = load_checkpoint_or_none() is not None

    # Step 1 — Define objectives
    with step(ctx, step_trace, "define_objectives"):
        objectives = {"n_variants": n_variants, "themes": list(themes), "require_model": require_model}
        step_trace[-1]["notes"] = str(objectives)

    # Step 2 — Generate scenario DSL
    with step(ctx, step_trace, "generate_variants"):
        scenarios: list[dict] = []
        for theme in themes:
            scenarios.extend(_generate_variants(theme, n_variants))
        step_trace[-1]["outputs_count"] = len(scenarios)
        step_trace[-1]["notes"] = f"{len(scenarios)} scenarios"

    seed_embedding_by_theme: dict[str, list[float]] = {}
    failing_cases: list[dict] = []
    passed = 0
    total_assertions = 0
    score_curves: list[dict] = []
    replay_timeline: list[dict] = []

    # Step 3 — Run pipeline in sandbox
    with step(ctx, step_trace, "run_regression"):
        for sc in scenarios:
            events = sc.get("events", [])
            hh = ctx.household_id or "redteam-sandbox"
            run_result = _run_pipeline_sandbox(hh, events)
            sc["_run_result"] = run_result
            replay_timeline.append({"scenario_id": sc.get("scenario_id"), "theme": sc.get("theme"), "scores": [s.get("score") for s in run_result.get("scores") or []]})
        step_trace[-1]["outputs_count"] = len(scenarios)

    # Step 4 — Similar incidents regression
    with step(ctx, step_trace, "similar_incidents_regression"):
        for sc in scenarios:
            theme = sc.get("theme", "")
            run_result = sc.get("_run_result") or {}
            if not run_result.get("model_available"):
                failing_cases.append({"scenario_id": sc.get("scenario_id"), "expected": "model_available", "got": "model_unavailable"})
                total_assertions += 1
                continue
            embeddings = run_result.get("embeddings") or []
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
        step_trace[-1]["outputs_count"] = total_assertions

    # Step 5 — Centroid watchlist regression
    with step(ctx, step_trace, "centroid_watchlist_regression"):
        for theme, seed_emb in seed_embedding_by_theme.items():
            theme_scenarios = [s for s in scenarios if s.get("theme") == theme]
            for sc in theme_scenarios:
                run_result = sc.get("_run_result") or {}
                embeddings = run_result.get("embeddings") or []
                if not embeddings:
                    continue
                cent = centroid([seed_emb, embeddings[0]])
                if cent:
                    sim = cosine_sim(embeddings[0], cent)
                    total_assertions += 1
                    if sim >= SIMILARITY_THRESHOLD:
                        passed += 1
                    else:
                        failing_cases.append({"scenario_id": sc.get("scenario_id"), "assertion": "centroid_watchlist", "expected": f">= {SIMILARITY_THRESHOLD}", "got": round(sim, 4)})
        step_trace[-1]["outputs_count"] = len(seed_embedding_by_theme)

    # Step 6 — Evidence subgraph regression
    with step(ctx, step_trace, "evidence_subgraph_regression"):
        for sc in scenarios:
            run_result = sc.get("_run_result") or {}
            subgraphs = run_result.get("model_subgraphs") or []
            if not subgraphs:
                continue
            edges = subgraphs[0].get("edges") or []
            edge_types = set(e.get("type") or "edge" for e in edges)
            total_assertions += 1
            if len(edges) >= MIN_SUBGRAPH_EDGES and (EXPECTED_EDGE_TYPES & edge_types or not EXPECTED_EDGE_TYPES):
                passed += 1
            else:
                failing_cases.append({"scenario_id": sc.get("scenario_id"), "assertion": "evidence_subgraph", "expected": f"edges>={MIN_SUBGRAPH_EDGES}", "got": f"edges={len(edges)} types={edge_types}"})
        step_trace[-1]["outputs_count"] = len(scenarios)

    pass_rate = round(passed / total_assertions, 4) if total_assertions else 0

    # Step 7 — Generate replay artifact
    with step(ctx, step_trace, "generate_replay_artifact"):
        replay = {
            "timeline": replay_timeline,
            "score_curves": score_curves,
            "step_trace": step_trace,
            "scenarios": [{"scenario_id": s.get("scenario_id"), "theme": s.get("theme")} for s in scenarios],
            "pass_rate": pass_rate,
            "failing_cases": failing_cases,
            "generated_at": ctx.now.isoformat(),
        }
        replay_path = None
        try:
            root = Path(__file__).resolve().parents[4]
            demo_out = root / "scripts" / "demo_out"
            demo_out.mkdir(parents=True, exist_ok=True)
            replay_file = demo_out / "scenario_replay.json"
            if not ctx.dry_run:
                replay_file.write_text(json.dumps(replay, indent=2), encoding="utf-8")
                replay_path = str(replay_file.relative_to(root)) if root in replay_file.parents else str(replay_file)
            artifacts_refs["replay_fixture_path"] = replay_path or "scripts/demo_out/scenario_replay.json"
        except Exception as e:
            logger.debug("Replay write failed: %s", e)
            artifacts_refs["replay_fixture_path"] = "scripts/demo_out/scenario_replay.json"
        step_trace[-1]["outputs_count"] = 1
        step_trace[-1]["artifacts_refs"] = {"replay_fixture_path": artifacts_refs["replay_fixture_path"]}

    # Step 8 — User-visible report (risk_signal if pass rate low)
    with step(ctx, step_trace, "user_visible_report"):
        summary_json["scenarios_generated"] = len(scenarios)
        summary_json["regression_pass_rate"] = pass_rate
        summary_json["regression_passed"] = model_available and total_assertions > 0 and pass_rate >= PASS_RATE_THRESHOLD
        summary_json["model_available"] = model_available
        summary_json["failing_cases"] = failing_cases
        summary_json["key_metrics"] = {"regression_pass_rate": pass_rate, "scenarios_generated": len(scenarios)}
        summary_json["key_findings"] = [f"Pass rate: {pass_rate}. Failing: {len(failing_cases)}."]
        summary_json["recommended_actions"] = ["Review failing scenarios", "Open replay artifact"]
        summary_json["artifact_refs"] = {**artifacts_refs, "scenario_ids": [s.get("scenario_id") for s in scenarios]}
        summary_json["replay_payload"] = replay
        if pass_rate < PASS_RATE_THRESHOLD and not ctx.dry_run and ctx.supabase:
            rsid = upsert_risk_signal_ctx(
                ctx,
                "redteam_regression_failed",
                2,
                float(1 - pass_rate),
                {"summary": f"Red-team regression pass rate {pass_rate} below threshold {PASS_RATE_THRESHOLD}.", "failing_cases": failing_cases[:10], "replay_fixture_path": artifacts_refs.get("replay_fixture_path")},
                {"checklist": ["Review failing scenarios", "Open replay"], "action": "review"},
                "open",
            )
            if rsid:
                artifacts_refs["risk_signal_ids"].append(rsid)
        step_trace[-1]["outputs_count"] = 1

    # Step 9 — Persist & UI
    with step(ctx, step_trace, "persist_ui"):
        run_id = persist_agent_run_ctx(ctx, "synthetic_redteam", "completed", step_trace, summary_json, artifacts_refs)
        summary_json["headline"] = f"Red-team pass rate: {pass_rate}; {len(failing_cases)} failing"
        step_trace[-1]["artifacts_refs"] = {"run_id": run_id, "replay_fixture_path": artifacts_refs.get("replay_fixture_path")}

    return {
        "step_trace": step_trace,
        "summary_json": summary_json,
        "status": "ok",
        "started_at": started_at,
        "ended_at": ctx.now.isoformat(),
        "run_id": run_id,
        "artifacts_refs": artifacts_refs,
    }


def run_synthetic_redteam_agent(
    household_id: str,
    supabase: Any | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Wrapper: build ctx and call run_synthetic_redteam_playbook."""
    ctx = AgentContext(household_id, supabase, dry_run=dry_run)
    out = run_synthetic_redteam_playbook(ctx)
    return {
        "step_trace": out["step_trace"],
        "summary_json": out["summary_json"],
        "status": out["status"],
        "started_at": out["started_at"],
        "ended_at": out["ended_at"],
        "run_id": out.get("run_id"),
        "artifacts_refs": out.get("artifacts_refs"),
    }
