"""
Single risk scoring service: one function, one response schema.
Returns raw_score, calibrated_p, rule_score, fusion_score, uncertainty; uses Platt end-to-end.
When model unavailable, returns model_available=False and scores=[] (callers use rule-only fallback via domain.rule_scoring).
"""
from __future__ import annotations

import logging
import math
from typing import Any

from api.schemas import RiskScoreItem, RiskScoringModelMeta, RiskScoringResponse
from domain.explainers.pg_service import attach_pg_explanations
from domain.rule_scoring import compute_rule_score

logger = logging.getLogger(__name__)


def _default_checkpoint_path():
    from pathlib import Path
    try:
        from config.settings import get_ml_settings
        return Path(get_ml_settings().checkpoint_path)
    except Exception:
        return Path("runs/hgt_baseline/best.pt")


def _pipeline_settings():
    try:
        from config.settings import get_pipeline_settings
        return get_pipeline_settings()
    except ImportError:
        class _F:
            explanation_score_min: float = 0.4
        return _F()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))


def score_risk(
    household_id: str,
    *,
    sessions: list[dict],
    utterances: list[dict],
    entities: list[dict],
    mentions: list[dict],
    relationships: list[dict],
    devices: list[dict] | None = None,
    events: list[dict] | None = None,
    checkpoint_path: Any = None,
    explanation_score_min: float | None = None,
    calibration_params: dict[str, Any] | None = None,
    pattern_tags: list[str] | None = None,
    structural_motifs: list[dict] | None = None,
) -> RiskScoringResponse:
    """
    Run GNN risk scoring; apply Platt calibration when calibration_params provided;
    fuse with rule_score when pattern_tags/structural_motifs provided.
    Returns scores with raw_score, calibrated_p, rule_score, fusion_score, uncertainty, decision_rule_used.
    """
    if not entities:
        return RiskScoringResponse(model_available=False, scores=[])

    devices = devices or []
    events = events or []
    path = checkpoint_path
    if path is None:
        path = _default_checkpoint_path()

    # Phase 4: model registry — use ANCHOR_GNN_MODEL when set
    raw_scores: list[dict] = []
    target_node_type: str | None = "entity"
    model = None
    data = None
    device = None
    try:
        from ml.registry import get_runner, get_model_name
        runner = get_runner(get_model_name(), checkpoint_path=path)
        if runner:
            rs, tn = runner.run(
                household_id,
                sessions=sessions,
                utterances=utterances,
                entities=entities,
                mentions=mentions,
                relationships=relationships,
                devices=devices,
                events=events,
            )
            if rs:
                raw_scores = rs
                target_node_type = tn or "entity"
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Registry runner failed: %s", e)

    # Fallback: inline load_model + run_inference (when registry not used or returned no scores)
    if not raw_scores:
        if hasattr(path, "is_file") and not path.is_file():
            logger.debug("Checkpoint missing at %s", path)
            return RiskScoringResponse(model_available=False, scores=[])
        try:
            import torch
            from ml.inference import load_model, run_inference
            from ml.graph.builder import build_hetero_from_tables
        except ImportError as e:
            logger.debug("ML stack unavailable: %s", e)
            return RiskScoringResponse(model_available=False, scores=[])
        try:
            device = torch.device("cpu")
            model, target_node_type = load_model(path, device)
            data = build_hetero_from_tables(
                household_id,
                sessions,
                utterances,
                entities,
                mentions,
                relationships,
                devices=devices,
            )
            raw_scores, _ = run_inference(
                model, data, device,
                target_node_type=target_node_type,
                return_embeddings=True,
            )
        except Exception as e:
            logger.debug("Inference failed: %s", e)
            return RiskScoringResponse(model_available=False, scores=[])

    # Attach PGExplainer only when we have model + data (fallback path); registry path skips explainer
    expl_min = explanation_score_min if explanation_score_min is not None else getattr(_pipeline_settings(), "explanation_score_min", 0.4)
    if model is not None and data is not None and device is not None:
        try:
            from config.settings import get_agent_settings
            top_k_edges = getattr(get_agent_settings(), "risk_scoring_top_k_edges", 20)
        except Exception:
            top_k_edges = 20
        attach_pg_explanations(
            model, data, raw_scores, entities,
            target_node_type=target_node_type or "entity",
            explanation_score_min=expl_min,
            top_k_edges=top_k_edges,
            device=device,
        )

    # Platt calibration
    platt_a = calibration_params.get("platt_a") if calibration_params else None
    platt_b = calibration_params.get("platt_b") if calibration_params else None
    conformal_q_hat = calibration_params.get("conformal_q_hat") if calibration_params else None
    coverage_level = calibration_params.get("coverage_level") if calibration_params else None
    calibration_size = calibration_params.get("calibration_size") if calibration_params else None
    for r in raw_scores:
        raw = float(r.get("score", 0))
        r["raw_score"] = raw
        if platt_a is not None and platt_b is not None:
            cal_p = _sigmoid(platt_a * raw + platt_b)
            r["calibrated_p"] = round(cal_p, 4)
            r["decision_rule_used"] = "conformal" if conformal_q_hat is not None else "calibrated"
        else:
            r["calibrated_p"] = None
            r["decision_rule_used"] = "raw_threshold"
        r["uncertainty"] = round(0.1 if r.get("calibrated_p") is not None else 0.2, 4)

    # Rule score per entity (for fusion or when model unavailable we don't call this path; pipeline does rule-only)
    pattern_tags = pattern_tags or []
    structural_motifs = structural_motifs or []
    rule_scores: list[float] = []
    if pattern_tags or structural_motifs:
        for i, e in enumerate(entities):
            entity_meta = {}
            if e:
                entity_meta["bridges_independent_sets"] = e.get("bridges_independent_sets", False)
                if e.get("independence_violation_ratio") is not None:
                    entity_meta["independence_violation_ratio"] = e.get("independence_violation_ratio")
            rule_scores.append(compute_rule_score(pattern_tags, structural_motifs, entity_meta or None))
    else:
        rule_scores = [None] * len(entities)

    # Fusion and final score. Fusion weight 0.6/0.4 is heuristic and explicit;
    # learnable fusion or logistic regression on (rule_score, calibrated_p) is future work.
    for i, r in enumerate(raw_scores):
        cal_p = r.get("calibrated_p")
        rule_s = rule_scores[i] if i < len(rule_scores) else None
        r["rule_score"] = round(rule_s, 4) if rule_s is not None else None
        if cal_p is not None and rule_s is not None:
            fusion = 0.6 * cal_p + 0.4 * rule_s
            r["fusion_score"] = round(fusion, 4)
            r["score"] = round(fusion, 4)
        elif cal_p is not None:
            r["fusion_score"] = None
            r["score"] = round(cal_p, 4)
        else:
            r["fusion_score"] = None
            r["score"] = round(r["raw_score"], 4)

    # Conformal risk bands: decision tiers for UI and escalation
    for r in raw_scores:
        cal_p = r.get("calibrated_p")
        unc = r.get("uncertainty") or 0.2
        if conformal_q_hat is not None and cal_p is not None and (1.0 - cal_p) >= conformal_q_hat:
            r["risk_band"] = "guaranteed_risk"
        elif cal_p is not None:
            if cal_p >= 0.7 and unc < 0.2:
                r["risk_band"] = "high_confidence"
            elif cal_p >= 0.5 or unc < 0.25:
                r["risk_band"] = "medium_confidence"
            else:
                r["risk_band"] = "low_confidence"
        else:
            r["risk_band"] = "low_confidence"

    scores_list: list[RiskScoreItem] = []
    embedding_dim: int | None = None
    for r in raw_scores:
        emb = r.get("embedding") if isinstance(r.get("embedding"), (list, tuple)) else None
        if emb and embedding_dim is None:
            embedding_dim = len(emb)
        item = RiskScoreItem(
            node_type=r.get("node_type", "entity"),
            node_index=r["node_index"],
            score=r["score"],
            signal_type=r.get("signal_type", "relational_anomaly"),
            embedding=emb,
            model_subgraph=r.get("model_subgraph"),
            model_available=True,
            raw_score=r.get("raw_score"),
            calibrated_p=r.get("calibrated_p"),
            rule_score=r.get("rule_score"),
            fusion_score=r.get("fusion_score"),
            uncertainty=r.get("uncertainty"),
            decision_rule_used=r.get("decision_rule_used"),
            risk_band=r.get("risk_band"),
        )
        scores_list.append(item)

    model_name = "hgt_baseline"
    try:
        from config.settings import get_ml_settings
        model_name = getattr(get_ml_settings(), "model_version_tag", None) or model_name
    except Exception:
        pass
    model_meta = RiskScoringModelMeta(
        model_name=model_name,
        checkpoint_id=str(path) if path else None,
        embedding_dim=embedding_dim,
        conformal_q_hat=conformal_q_hat,
        coverage_level=coverage_level,
        calibration_size=calibration_size,
    )
    return RiskScoringResponse(model_available=True, scores=scores_list, model_meta=model_meta)
