"""Graph domain: normalize events to utterances, entities, mentions, relationships."""
from typing import Any

from ml.graph.builder import GraphBuilder


def normalize_events(
    household_id: str,
    events: list[dict[str, Any]],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Build graph structures from raw events using GraphBuilder.
    Returns (utterances, entities, mentions, relationships).
    """
    builder = GraphBuilder(household_id)
    by_session: dict[str, list] = {}
    for ev in events:
        sid = ev.get("session_id") or ""
        if isinstance(sid, dict):
            sid = sid.get("id", "")
        sid = str(sid)
        if sid not in by_session:
            by_session[sid] = []
        by_session[sid].append(ev)
    for sid, evs in by_session.items():
        device_id = evs[0].get("device_id", "") if evs else ""
        if isinstance(device_id, dict):
            device_id = device_id.get("id", "")
        builder.process_events(evs, sid, str(device_id))
    return (
        builder.get_utterance_list(),
        builder.get_entity_list(),
        builder.get_mention_list(),
        builder.get_relationship_list(),
    )
