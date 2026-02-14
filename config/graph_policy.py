"""
Graph mutation policy: failure containment.

No graph mutation (utterances, entities, mentions) when ASR or intent confidence
is below threshold. Events are still ingested to Supabase; only the derived
graph build is gated. Contract: README_EXTENDED.md §3.2, event_packet_spec.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _get_confidence_min() -> float:
    try:
        from config.settings import get_pipeline_settings
        return get_pipeline_settings().asr_confidence_min_for_graph
    except Exception:
        return 0.0


def allow_graph_mutation_for_event(event: dict[str, Any], *, confidence_min: float | None = None) -> bool:
    """
    Return True if this event is allowed to mutate the graph (add utterances,
    entities, mentions). False when ASR or intent confidence is below threshold
    (failure containment: do not propagate low-confidence data into the graph).

    Events that do not carry confidence (wake, tool_result, device_state, etc.)
    are allowed when they are not confidence-gated types. final_asr and intent
    are gated by payload.confidence.
    """
    confidence_min = confidence_min if confidence_min is not None else _get_confidence_min()
    if confidence_min <= 0.0:
        return True

    event_type = (event.get("event_type") or "").strip()
    payload = event.get("payload") or {}

    if event_type == "final_asr":
        conf = payload.get("confidence")
        if conf is None:
            # Missing confidence: treat as low, do not mutate (fail safe).
            logger.debug(
                "event_policy: skipping graph mutation for final_asr with no confidence",
                extra={"session_id": event.get("session_id"), "seq": event.get("seq")},
            )
            return False
        if conf < confidence_min:
            logger.debug(
                "event_policy: skipping graph mutation for low ASR confidence %.3f < %.3f",
                conf,
                confidence_min,
                extra={"session_id": event.get("session_id"), "seq": event.get("seq")},
            )
            return False
        return True

    if event_type == "intent":
        conf = payload.get("confidence")
        if conf is None:
            # Intent without confidence: treat as low (fail safe).
            logger.debug(
                "event_policy: skipping graph mutation for intent with no confidence",
                extra={"session_id": event.get("session_id"), "seq": event.get("seq")},
            )
            return False
        if conf < confidence_min:
            logger.debug(
                "event_policy: skipping graph mutation for low intent confidence %.3f < %.3f",
                conf,
                confidence_min,
                extra={"session_id": event.get("session_id"), "seq": event.get("seq")},
            )
            return False
        return True

    # wake, partial_asr, tool_call, tool_result, tts, error, device_state, watchlist_hit: no confidence gate.
    return True
