"""Claude-generated title and narrative for risk signals from agent outputs (motifs, timeline, evidence)."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def generate_risk_signal_title_and_narrative(
    signal_type: str,
    severity: int,
    explanation: dict[str, Any],
) -> dict[str, str] | None:
    """
    Use Claude to generate a short title and a 2–4 sentence narrative for a risk signal
    based on motif_tags, timeline_snippet, what_changed_summary, and other agent outputs.
    Returns {"title": "...", "narrative": "..."} or None if API key missing or call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    motifs = explanation.get("motif_tags") or explanation.get("semantic_pattern_tags") or []
    what_changed = (explanation.get("what_changed_summary") or "")[:500]
    timeline = explanation.get("timeline_snippet") or []
    timeline_preview = ""
    for i, item in enumerate(timeline[:5]):
        if isinstance(item, dict):
            timeline_preview += (item.get("text_preview") or item.get("summary") or str(item))[:150] + " "
        else:
            timeline_preview += str(item)[:150] + " "
    subgraph = explanation.get("subgraph") or explanation.get("model_subgraph") or {}
    nodes = subgraph.get("nodes") or []
    entity_preview = ", ".join(str(n.get("id") or n.get("label") or "") for n in nodes[:5] if isinstance(n, dict))[:200]
    context = (
        f"Signal type: {signal_type}. Severity (1-5): {severity}.\n"
        f"Motifs/tags: {', '.join(motifs[:10]) if motifs else 'None'}.\n"
        f"What changed: {what_changed or 'N/A'}.\n"
    )
    if timeline_preview.strip():
        context += f"Timeline snippet: {timeline_preview.strip()[:400]}.\n"
    if entity_preview:
        context += f"Entities involved: {entity_preview}.\n"

    prompt = f"""You are writing for an elder-safety app (Anchor). Generate a brief, clear title and narrative for this risk signal. Use ONLY the facts below. Do not invent details.

{context}

Respond with a JSON object only, no other text:
{{
  "title": "Short headline (max 80 chars). Example: 'Repeated contact from unknown number with Medicare-themed language'",
  "narrative": "2-4 sentences describing what happened and why it may be concerning. Use the timeline and motifs. Max 400 characters."
}}

Rules: title under 80 characters. narrative under 400 characters. Be specific to the evidence. Tone: factual, not alarming."""

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in msg.content:
            if hasattr(block, "text"):
                text += block.text
        text = text.strip()
        if not text:
            return None
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        title = (data.get("title") or "").strip()[:80]
        narrative = (data.get("narrative") or "").strip()[:500]
        if title or narrative:
            return {"title": title or "Risk signal", "narrative": narrative or what_changed or "Activity may need review."}
        return None
    except Exception as e:
        logger.warning("Claude risk narrative failed: %s", e)
        return None
