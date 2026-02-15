"""
Layer A: Semantic pattern tags (rule-based) + structural motif detection (graph patterns).
Semantic: new_contact+urgency, bursty_contact, device_switch, contact→intent(sensitive) cascade.
Structural: triadic closure with urgency edges, 2-hop chain, star pattern (subgraph isomorphism).
Explanation has semantic_pattern_tags and structural_motifs; "motif" is structurally grounded.

Future directions for a defensible academic system: temporal motifs (time-ordered subgraphs),
repeated motif recurrence, and motif novelty relative to baseline (integrates with drift
and independence: novelty signals distribution shift).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def _ts_float(ts: Any) -> float:
    try:
        from domain.utils.time_utils import ts_to_float
        return ts_to_float(ts)
    except ImportError:
        if ts is None:
            return 0.0
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            from datetime import datetime, timezone
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return t.timestamp() if t.tzinfo else t.replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                return 0.0
        if hasattr(ts, "timestamp"):
            return ts.timestamp()
        return 0.0


def _motif_keywords() -> tuple[frozenset[str], frozenset[str]]:
    try:
        from config.graph import get_graph_config
        g = get_graph_config()
        return g.get("urgency_topics") or frozenset(), g.get("sensitive_intents") or frozenset()
    except ImportError:
        urgency = frozenset({"medicare", "irs", "social security", "ssn", "urgent", "immediately", "suspended", "account", "verify", "confirm"})
        sensitive = frozenset({"sensitive_request", "share_ssn", "share_card", "pay_now", "wire_money", "buy_gift_cards"})
        return urgency, sensitive


def _detect_structural_motifs(
    entities: list[dict],
    relationships: list[dict],
    mentions: list[dict],
    entity_first_ts: dict[str, float],
    urgency_entity_ids: set[str],
) -> list[dict]:
    """
    Detect structural motifs via subgraph patterns: triadic closure with urgency,
    2-hop chain (unknown_contact → urgency_topic → sensitive_intent), star (one entity contacts many in short window).
    Returns list of {pattern_type, node_ids, subgraph_node_ids}.
    """
    structural_motifs: list[dict] = []
    try:
        import networkx as nx
    except ImportError:
        return structural_motifs
    entity_ids = [e["id"] for e in entities if e.get("id")]
    if len(entity_ids) < 2:
        return structural_motifs
    G = nx.Graph()
    G.add_nodes_from(entity_ids)
    for r in relationships:
        src, dst = r.get("src_entity_id"), r.get("dst_entity_id")
        if src and dst and src in entity_ids and dst in entity_ids:
            G.add_edge(src, dst)

    # Star: one node with degree >= 2 (one entity co-occurs with many others in short window)
    for n in G.nodes():
        if G.degree(n) >= 2:
            neighbors = list(G.neighbors(n))
            structural_motifs.append({
                "pattern_type": "star_pattern",
                "node_ids": [n] + neighbors[:5],
                "subgraph_node_ids": [n] + neighbors,
                "center_id": n,
                "degree": G.degree(n),
            })
            break

    # Triadic closure: triangle; tag if any node is in urgency set
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) == 3:
            nodes = list(clique)
            if any(n in urgency_entity_ids for n in nodes):
                structural_motifs.append({
                    "pattern_type": "triadic_closure_urgency",
                    "node_ids": nodes,
                    "subgraph_node_ids": nodes,
                })
            break

    # 2-hop chain: A-B-C where B is urgency-related (simplified: any 2-hop path)
    if len(entity_ids) >= 3:
        for src in entity_ids[:10]:
            for mid in G.neighbors(src):
                for dst in G.neighbors(mid):
                    if dst != src and not G.has_edge(src, dst):
                        structural_motifs.append({
                            "pattern_type": "2hop_chain",
                            "node_ids": [src, mid, dst],
                            "subgraph_node_ids": [src, mid, dst],
                        })
                        break
                else:
                    continue
                break

    return structural_motifs[:10]


def extract_motifs(
    utterances: list[dict],
    mentions: list[dict],
    entities: list[dict],
    relationships: list[dict],
    events: list[dict],
    entity_id_to_canonical: dict[str, str] | None = None,
    urgency_topics: set[str] | frozenset[str] | None = None,
    sensitive_intents: set[str] | frozenset[str] | None = None,
) -> tuple[list[str], list[dict], list[dict]]:
    """
    Returns (semantic_pattern_tags: list[str], timeline_snippet: list[dict], structural_motifs: list[dict]).
    semantic_pattern_tags are plain-English labels; structural_motifs are graph patterns with pattern_type and node_ids.
    """
    u_topics = urgency_topics if urgency_topics is not None else _motif_keywords()[0]
    s_intents = sensitive_intents if sensitive_intents is not None else _motif_keywords()[1]
    motif_tags: list[str] = []
    timeline_snippet: list[dict] = []
    entity_id_to_canonical = entity_id_to_canonical or {}
    # Build entity first-seen ts and type
    entity_first_ts: dict[str, float] = {}
    entity_type: dict[str, str] = {}
    for e in entities:
        entity_type[e["id"]] = e.get("entity_type", "topic")
    for m in mentions:
        eid = m.get("entity_id")
        ts = _ts_float(m.get("ts"))
        if eid and (eid not in entity_first_ts or ts < entity_first_ts[eid]):
            entity_first_ts[eid] = ts

    # 1) New contact + urgency topic
    seen_entities_this_session: set[str] = set()
    for u in sorted(utterances, key=lambda x: _ts_float(x.get("ts"))):
        utt_ts = _ts_float(u.get("ts"))
        text = (u.get("text") or "").lower()
        intent = (u.get("intent") or "").lower()
        for m in mentions:
            if m.get("utterance_id") != u.get("id"):
                continue
            eid = m.get("entity_id")
            if eid in seen_entities_this_session:
                continue
            seen_entities_this_session.add(eid)
            canonical = entity_id_to_canonical.get(eid, "")
            is_new = entity_first_ts.get(eid, float("inf")) >= utt_ts - 3600  # first seen in last hour
            if is_new and any(t in text or t in intent for t in u_topics):
                motif_tags.append("New contact + urgency topic (e.g. Medicare/IRS)")
                timeline_snippet.append({"ts": utt_ts, "type": "utterance", "motif": "new_contact_urgency", "text_preview": text[:80]})
                break

    # 2) Bursty repeated contact attempts
    contact_counts: dict[str, list[float]] = defaultdict(list)
    for m in mentions:
        eid = m.get("entity_id")
        if entity_type.get(eid) in ("phone", "person"):
            contact_counts[eid].append(_ts_float(m.get("ts")))
    for eid, ts_list in contact_counts.items():
        if len(ts_list) >= 3:
            # multiple mentions in same session
            motif_tags.append("Bursty repeated contact attempts")
            timeline_snippet.append({"ts": ts_list[0], "type": "mention_burst", "entity_id": eid, "count": len(ts_list)})
            break

    # 3) Device switching (if we have device_id per event)
    device_per_ts: list[tuple[float, str]] = []
    for ev in events:
        device_id = ev.get("device_id") or ""
        device_per_ts.append((_ts_float(ev.get("ts")), str(device_id)))
    if len(set(d for _, d in device_per_ts)) > 1 and len(device_per_ts) >= 2:
        motif_tags.append("Device switching during session")

    # 4) Cascade: contact → intent(sensitive) → new payee attempt
    intents_seen: list[tuple[float, str]] = []
    for u in utterances:
        if u.get("intent"):
            intents_seen.append((_ts_float(u.get("ts")), (u.get("intent") or "").lower()))
    has_sensitive = any(any(s in i for s in s_intents) for _, i in intents_seen)
    has_new_contact = len(seen_entities_this_session) > 0
    if has_new_contact and has_sensitive:
        motif_tags.append("Contact → sensitive intent (e.g. share info / pay) cascade")
        for ts, i in intents_seen:
            if any(s in i for s in s_intents):
                timeline_snippet.append({"ts": ts, "type": "intent", "intent": i, "motif": "sensitive_intent"})
                break

    # Structural motifs (entity graph patterns)
    urgency_entity_ids = set()
    for eid in entity_first_ts:
        if entity_type.get(eid) == "topic":
            urgency_entity_ids.add(eid)
    structural_motifs = _detect_structural_motifs(
        entities, relationships, mentions, entity_first_ts, urgency_entity_ids,
    )

    # Dedupe and cap timeline
    seen_motif = set()
    motif_tags_dedup = []
    for t in motif_tags:
        if t not in seen_motif:
            seen_motif.add(t)
            motif_tags_dedup.append(t)
    timeline_snippet = sorted(timeline_snippet, key=lambda x: x.get("ts", 0))[:6]
    return motif_tags_dedup, timeline_snippet, structural_motifs
