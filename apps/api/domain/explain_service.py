"""
Explain opaque identifiers (entity IDs, alert IDs, etc.) in plain language for caregivers.
Uses Claude when ANTHROPIC_API_KEY is set; returns generic, safe descriptions (no PII).

Also: build_subgraph_from_explanation and run_deep_dive_explainer for risk-signal explanation subgraphs.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import UUID

from supabase import Client

from api.schemas import RiskSignalDetailSubgraph, SubgraphEdge, SubgraphNode

logger = logging.getLogger(__name__)

EXPLAIN_CONTEXTS = {
    "pattern_members": "Items in a detected pattern (e.g. people, phone numbers, topics we linked together).",
    "alert_ids": "References to risk alerts we created (each alert is one detected concern).",
    "top_connectors": "Graph nodes that connect many others in the pattern (often key contacts or numbers).",
    "entity_list": "Entities in the household graph (contacts, devices, topics, or intents we detected).",
}


def explain_opaque_items(
    context: str,
    items: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    For each item (with 'id' and optional 'hint'), return a short plain-English explanation.
    items: [{"id": "acc_xyz", "hint": "optional type or label"}]
    Returns: [{"original": "acc_xyz", "explanation": "..."}]
    """
    if not items:
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.debug("ANTHROPIC_API_KEY not set; returning fallback explanations")
        return _fallback_explanations(context, items)

    context_desc = EXPLAIN_CONTEXTS.get(
        context,
        "Internal identifiers shown in the app that we want to explain in plain language.",
    )
    ids_list = "\n".join(
        f"- {it.get('id', '')}" + (f" (hint: {it.get('hint', '')})" if it.get("hint") else "")
        for it in items[:20]
    )

    prompt = f"""We show users these opaque identifiers in our caregiving/safety app. For each one, write exactly one short sentence (under 100 characters) that explains what kind of thing it is and how it relates to this context. Do not guess real names or personal details. Be caregiver-friendly and consistent.

Context: {context_desc}

Identifiers:
{ids_list}

Respond with a JSON array of objects, one per identifier in the same order: [{{"original": "<id>", "explanation": "<one sentence>"}}, ...]. No other text."""

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in msg.content:
            if hasattr(block, "text"):
                text += block.text
        if not text.strip():
            return _fallback_explanations(context, items)
        parsed = json.loads(text.strip())
        if isinstance(parsed, list) and len(parsed) >= len(items):
            return parsed[: len(items)]
        return _fallback_explanations(context, items)
    except Exception as e:
        logger.warning("Explain (Claude) failed: %s", e)
        return _fallback_explanations(context, items)


def _fallback_explanations(context: str, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Deterministic fallbacks when Claude is unavailable."""
    out = []
    for it in items:
        orig = str(it.get("id", ""))
        if context == "pattern_members":
            expl = "Part of this pattern — a contact, number, or topic we linked."
        elif context == "alert_ids":
            expl = "A risk alert we created; open it to see what we found and what to do."
        elif context == "top_connectors":
            expl = "A key node in this pattern that connects to several others."
        else:
            expl = "An internal reference in our system; open the linked page for details."
        out.append({"original": orig, "explanation": expl})
    return out


def build_subgraph_from_explanation(
    expl: dict[str, Any],
    prefer_key: str | None = None,
) -> RiskSignalDetailSubgraph | None:
    """Build RiskSignalDetailSubgraph from explanation dict. Prefer model_subgraph, or subgraph, or prefer_key (e.g. deep_dive_subgraph)."""
    raw = None
    if prefer_key and expl.get(prefer_key):
        raw = expl[prefer_key]
    if raw is None:
        raw = expl.get("model_subgraph") or expl.get("subgraph")
    if not raw or not isinstance(raw, dict):
        return None
    nodes_raw = raw.get("nodes") or []
    edges_raw = raw.get("edges") or []
    if not nodes_raw and not edges_raw:
        return None
    nodes = [
        SubgraphNode(
            id=n.get("id", ""),
            type=n.get("type", ""),
            label=n.get("label"),
            score=n.get("score"),
        )
        for n in nodes_raw
    ]
    edges = [
        SubgraphEdge(
            src=e.get("src", ""),
            dst=e.get("dst", ""),
            type=e.get("type", ""),
            weight=e.get("weight") if "weight" in e else e.get("importance"),
            rank=e.get("rank"),
        )
        for e in edges_raw
    ]
    return RiskSignalDetailSubgraph(nodes=nodes, edges=edges)


def run_deep_dive_explainer(
    signal_id: UUID,
    household_id: str,
    supabase: Client,
    mode: str = "pg",
) -> dict[str, Any]:
    """Run deep-dive explainer: mode=pg copies model_subgraph to deep_dive_subgraph and persists; mode=gnn raises NotImplementedError."""
    if mode == "gnn":
        raise NotImplementedError("gnn mode not implemented")
    if mode != "pg":
        raise ValueError(f"mode must be 'pg' or 'gnn', got {mode!r}")
    row = (
        supabase.table("risk_signals")
        .select("id, explanation")
        .eq("id", str(signal_id))
        .eq("household_id", household_id)
        .single()
        .execute()
    )
    if not row.data:
        raise ValueError("Risk signal not found")
    expl = (row.data.get("explanation") or {}) if isinstance(row.data.get("explanation"), dict) else {}
    model_sg = expl.get("model_subgraph")
    if not model_sg or (isinstance(model_sg, dict) and not (model_sg.get("nodes") or model_sg.get("edges"))):
        raise ValueError("No model subgraph in explanation for deep dive")
    deep_dive = {
        "method": "pg",
        "nodes": list(model_sg.get("nodes") or []),
        "edges": list(model_sg.get("edges") or []),
    }
    new_expl = {**expl, "deep_dive_subgraph": deep_dive}
    supabase.table("risk_signals").update({"explanation": new_expl}).eq("id", str(signal_id)).eq("household_id", household_id).execute()
    return {"ok": True, "method": "pg", "deep_dive_subgraph": deep_dive}


# Re-export for backward compatibility (risk_signals and tests use similarity_service directly or this)
from domain.similarity_service import get_similar_incidents  # noqa: E402
