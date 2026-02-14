"""
Independence Graph builder: event packets -> utterances, entities, mentions, relationships;
and relational tables -> PyG HeteroData with time-aware edges.

Node types, edge types, entity types, event types, and feature dims come from config.graph
to keep the implementation scalable and limit hardcoding.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def _graph_config() -> dict[str, Any]:
    try:
        from config.graph import get_graph_config
        return get_graph_config()
    except ImportError:
        return {
            "entity_type_map": {"phone": "phone", "email": "email", "person": "person", "org": "org", "merchant": "merchant", "topic": "topic", "account": "account", "device": "device", "location": "location"},
            "slot_to_entity": {"phone": "phone", "number": "phone", "email": "email", "name": "person", "person": "person", "merchant": "merchant"},
            "event_types": frozenset({"final_asr", "intent", "watchlist_hit"}),
            "speaker_roles": frozenset({"elder", "agent", "unknown"}),
            "co_occurrence_window_sec": 300,
            "base_feature_dims": {"person": 8, "device": 8, "session": 8, "utterance": 16, "intent": 8, "entity": 16},
            "time_encoding_dim": 8,
            "edge_attr_dim": 4,
            "person_ids": ["person_elder"],
        }


# Backward compatibility: default entity type map
ENTITY_TYPE_MAP = _graph_config()["entity_type_map"]


def _ts_to_float(ts: datetime | str) -> float:
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()


def _hash_canonical(s: str) -> str:
    return hashlib.sha256(s.strip().lower().encode()).hexdigest()[:16]


class GraphBuilder:
    """
    Builds derived tables (utterances, entities, mentions, relationships) from events
    and optionally produces PyG HeteroData from those tables.
    """

    def __init__(self, household_id: str):
        self.household_id = household_id
        self.utterances: list[dict[str, Any]] = []
        self.entities: dict[tuple[str, str], dict] = {}  # (entity_type, canonical_hash) -> entity
        self.mentions: list[dict[str, Any]] = []
        self.relationships: dict[tuple[str, str, str], dict] = {}  # (src_id, dst_id, rel_type) -> rel
        self._entity_id_by_type_hash: dict[tuple[str, str], str] = {}

    def _get_or_create_entity(
        self,
        entity_type: str,
        canonical: str,
        canonical_hash: str | None = None,
        meta: dict | None = None,
    ) -> str:
        h = canonical_hash or _hash_canonical(canonical)
        key = (entity_type, h)
        if key in self._entity_id_by_type_hash:
            return self._entity_id_by_type_hash[key]
        eid = f"entity_{entity_type}_{h}"
        self._entity_id_by_type_hash[key] = eid
        self.entities[(entity_type, h)] = {
            "id": eid,
            "household_id": self.household_id,
            "entity_type": entity_type,
            "canonical": canonical,
            "canonical_hash": h,
            "meta": meta or {},
        }
        return eid

    def process_events(
        self,
        events: list[dict[str, Any]],
        session_id: str,
        device_id: str,
        session_ts_start: datetime | None = None,
    ) -> None:
        """
        Process a batch of events for one session: extract utterances, intents,
        entities from payloads and build mentions / co-occurrence.
        """
        utterances_in_session: list[dict] = []
        entities_mentioned: list[tuple[str, datetime, float]] = []  # (entity_id, ts, confidence)
        intents_in_session: list[tuple[str, datetime, float]] = []

        gconf = _graph_config()
        event_types = gconf.get("event_types") or frozenset({"final_asr", "intent", "watchlist_hit"})
        speaker_roles = gconf.get("speaker_roles") or frozenset({"elder", "agent", "unknown"})
        slot_to_entity = gconf.get("slot_to_entity") or {"phone": "phone", "number": "phone", "email": "email", "name": "person", "person": "person", "merchant": "merchant"}
        window_sec = gconf.get("co_occurrence_window_sec", 300)

        for ev in sorted(events, key=lambda x: (x.get("ts") or "", x.get("seq", 0))):
            ts = ev.get("ts")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            payload = ev.get("payload") or {}
            event_type = ev.get("event_type", "")

            if event_type in event_types and event_type == "final_asr":
                text = payload.get("text") or ""
                text_hash = payload.get("text_hash") or (_hash_canonical(text) if text else None)
                speaker = (payload.get("speaker") or {}).get("role", "unknown")
                if speaker not in speaker_roles:
                    speaker = "unknown"
                uid = f"utt_{session_id}_{ev.get('seq', 0)}"
                self.utterances.append({
                    "id": uid,
                    "session_id": session_id,
                    "ts": ts,
                    "speaker": speaker,
                    "text": text or None,
                    "text_hash": text_hash,
                    "intent": None,
                    "confidence": payload.get("confidence"),
                    "event_id": ev.get("id"),
                })
                utterances_in_session.append({"id": uid, "ts": ts})

            if event_type in event_types and event_type == "intent":
                name = (payload.get("name") or "").strip() or "unknown"
                confidence = payload.get("confidence", 0.0)
                intents_in_session.append((name, ts, confidence))
                # Link last utterance to this intent if we have one
                if utterances_in_session:
                    last_utt = utterances_in_session[-1]
                    self.utterances[-1]["intent"] = name
                    self.utterances[-1]["confidence"] = confidence

            # Extract entities from intent slots or payload (slot_to_entity from config)
            slots = payload.get("slots") or {}
            for slot_name, value in slots.items():
                if not value:
                    continue
                if isinstance(value, str):
                    slot_lower = slot_name.lower()
                    etype = "topic"
                    for key, entity_type in slot_to_entity.items():
                        if key in slot_lower:
                            etype = entity_type
                            break
                    eid = self._get_or_create_entity(etype, value)
                    entities_mentioned.append((eid, ts, confidence if event_type == "intent" else 0.5))

        # Co-occurrence: entities in same session window (window_sec from config)
        if entities_mentioned:
            by_window: dict[int, list[str]] = defaultdict(list)
            for eid, t, _ in entities_mentioned:
                bucket = int(_ts_to_float(t) // window_sec)
                by_window[bucket].append(eid)
            for bucket, eids in by_window.items():
                seen = set()
                for i, a in enumerate(eids):
                    for b in eids[i + 1 :]:
                        if a == b:
                            continue
                        pair = (min(a, b), max(a, b))
                        if pair in seen:
                            continue
                        seen.add(pair)
                        key = (pair[0], pair[1], "CO_OCCURS")
                        t_float = bucket * window_sec
                        if key not in self.relationships:
                            self.relationships[key] = {
                                "src_entity_id": pair[0],
                                "dst_entity_id": pair[1],
                                "rel_type": "CO_OCCURS",
                                "weight": 1.0,
                                "first_seen_at": t_float,
                                "last_seen_at": t_float,
                                "count": 0,
                                "evidence": [],
                            }
                        self.relationships[key]["count"] += 1
                        self.relationships[key]["last_seen_at"] = max(
                            self.relationships[key]["last_seen_at"], t_float
                        )

        # Mentions for this session
        for eid, ts, conf in entities_mentioned:
            self.mentions.append({
                "session_id": session_id,
                "entity_id": eid,
                "ts": ts,
                "confidence": conf,
                "utterance_id": utterances_in_session[-1]["id"] if utterances_in_session else None,
            })

        # watchlist_hit: device reports a watchlist match -> Entity —[TRIGGERED]→ evidence in graph
        if event_type in event_types and event_type == "watchlist_hit":
            payload_w = payload or {}
            entity_id = payload_w.get("entity_id") or payload_w.get("matched_entity_id")
            if entity_id:
                t_float = _ts_to_float(ts)
                key = (entity_id, entity_id, "TRIGGERED")
                ev_evidence = {"event_id": ev.get("id"), "watchlist_id": payload_w.get("watchlist_id")}
                if key not in self.relationships:
                    self.relationships[key] = {
                        "src_entity_id": entity_id,
                        "dst_entity_id": entity_id,
                        "rel_type": "TRIGGERED",
                        "weight": 1.0,
                        "first_seen_at": t_float,
                        "last_seen_at": t_float,
                        "count": 1,
                        "evidence": [ev_evidence],
                    }
                else:
                    self.relationships[key]["count"] += 1
                    self.relationships[key]["last_seen_at"] = max(self.relationships[key]["last_seen_at"], t_float)
                    self.relationships[key]["evidence"].append(ev_evidence)

    def get_entity_list(self) -> list[dict]:
        return list(self.entities.values())

    def get_utterance_list(self) -> list[dict]:
        return self.utterances

    def get_mention_list(self) -> list[dict]:
        return self.mentions

    def get_relationship_list(self) -> list[dict]:
        return list(self.relationships.values())


def build_hetero_from_tables(
    household_id: str,
    sessions: list[dict],
    utterances: list[dict],
    entities: list[dict],
    mentions: list[dict],
    relationships: list[dict],
    devices: list[dict] | None = None,
    time_encoding_dim: int | None = None,
    base_feature_dims: dict[str, int] | None = None,
    person_ids: list[str] | None = None,
    edge_attr_dim: int | None = None,
):
    """
    Build PyG HeteroData from normalized relational tables.
    Node types, feature dims, and time encoding from config when not passed.
    """
    import torch
    from torch_geometric.data import HeteroData
    from ml.graph.time_encoding import sinusoidal_time_encoding
    gconf = _graph_config()
    time_encoding_dim = time_encoding_dim if time_encoding_dim is not None else gconf.get("time_encoding_dim", 8)
    base = base_feature_dims if base_feature_dims is not None else gconf.get("base_feature_dims", {"person": 8, "device": 8, "session": 8, "utterance": 16, "intent": 8, "entity": 16})
    person_ids = person_ids if person_ids is not None else gconf.get("person_ids", ["person_elder"])
    edge_attr_dim = edge_attr_dim if edge_attr_dim is not None else gconf.get("edge_attr_dim", 4)

    data = HeteroData()
    devices = devices or []

    def _ts(ts_any) -> float:
        if ts_any is None:
            return 0.0
        if isinstance(ts_any, str):
            from datetime import datetime, timezone
            t = datetime.fromisoformat(ts_any.replace("Z", "+00:00"))
            return t.timestamp() if t.tzinfo else t.replace(tzinfo=timezone.utc).timestamp()
        if hasattr(ts_any, "timestamp"):
            return ts_any.timestamp()
        return float(ts_any)

    # Unique intents across utterances
    intents_set: set[str] = set()
    for u in utterances:
        if u.get("intent"):
            intents_set.add(u["intent"])
    intents_list = sorted(intents_set)
    intent_to_idx = {x: i for i, x in enumerate(intents_list)}

    session_ids = [s["id"] for s in sessions]
    utterance_ids = [u["id"] for u in utterances]
    entity_ids = [e["id"] for e in entities]
    device_ids = [d["id"] for d in devices] if devices else []

    # person_ids and base from config (or args)
    # Node timestamps for TGAT: utterance.ts, entity from first mention, session started_at
    utterance_ts = torch.tensor([_ts(u.get("ts")) for u in utterances], dtype=torch.float32)
    entity_first_ts: list[float] = []
    for e in entities:
        first = None
        for m in mentions:
            if m.get("entity_id") == e["id"]:
                t = _ts(m.get("ts"))
                if first is None or t < first:
                    first = t
        entity_first_ts.append(first if first is not None else 0.0)
    entity_ts = torch.tensor(entity_first_ts, dtype=torch.float32)
    session_ts = torch.tensor([_ts(s.get("started_at")) for s in sessions], dtype=torch.float32)

    # Node features = base + time_encoding (TGAT-style); base dims from config
    data["person"].x = torch.cat([torch.ones(len(person_ids), base.get("person", 8)), sinusoidal_time_encoding(torch.zeros(len(person_ids)), time_encoding_dim)], dim=1)
    if device_ids:
        data["device"].x = torch.cat([torch.ones(len(device_ids), base.get("device", 8)), sinusoidal_time_encoding(torch.zeros(len(device_ids)), time_encoding_dim)], dim=1)
    else:
        data["device"].x = torch.empty(0, base.get("device", 8) + time_encoding_dim)
    data["session"].x = torch.cat([torch.ones(len(session_ids), base.get("session", 8)), sinusoidal_time_encoding(session_ts, time_encoding_dim)], dim=1)
    data["utterance"].x = torch.cat([torch.ones(len(utterance_ids), base.get("utterance", 16)), sinusoidal_time_encoding(utterance_ts, time_encoding_dim)], dim=1)
    data["intent"].x = torch.cat([torch.ones(len(intents_list), base.get("intent", 8)), sinusoidal_time_encoding(torch.zeros(len(intents_list)), time_encoding_dim)], dim=1)
    data["entity"].x = torch.cat([torch.ones(len(entity_ids), base.get("entity", 16)), sinusoidal_time_encoding(entity_ts, time_encoding_dim)], dim=1)

    # Edges: (src_type, edge_type, dst_type)
    # Person -USES-> Device
    if device_ids:
        uses_src = [0] * len(device_ids)  # person_elder
        uses_dst = list(range(len(device_ids)))
        data["person", "uses", "device"].edge_index = torch.tensor([uses_src, uses_dst], dtype=torch.long)
        data["person", "uses", "device"].edge_attr = torch.zeros(len(device_ids), edge_attr_dim)

    # Session -HAS-> Utterance
    sess_to_idx = {s["id"]: i for i, s in enumerate(sessions)}
    utt_to_idx = {u["id"]: i for i, u in enumerate(utterances)}
    has_src, has_dst = [], []
    for u in utterances:
        sid = u.get("session_id")
        if sid in sess_to_idx:
            has_src.append(sess_to_idx[sid])
            has_dst.append(utt_to_idx[u["id"]])
    if has_src:
        data["session", "has", "utterance"].edge_index = torch.tensor([has_src, has_dst], dtype=torch.long)
        data["session", "has", "utterance"].edge_attr = torch.zeros(len(has_src), edge_attr_dim)

    # Utterance -EXPRESSES-> Intent
    expr_src, expr_dst = [], []
    for u in utterances:
        intent = u.get("intent")
        if intent and intent in intent_to_idx:
            expr_src.append(utt_to_idx[u["id"]])
            expr_dst.append(intent_to_idx[intent])
    if expr_src:
        data["utterance", "expresses", "intent"].edge_index = torch.tensor([expr_src, expr_dst], dtype=torch.long)
        data["utterance", "expresses", "intent"].edge_attr = torch.zeros(len(expr_src), edge_attr_dim)

    # Utterance -MENTIONS-> Entity
    entity_to_idx = {e["id"]: i for i, e in enumerate(entities)}
    ment_src, ment_dst = [], []
    for m in mentions:
        utt_id = m.get("utterance_id")
        eid = m.get("entity_id")
        if utt_id in utt_to_idx and eid in entity_to_idx:
            ment_src.append(utt_to_idx[utt_id])
            ment_dst.append(entity_to_idx[eid])
    if ment_src:
        data["utterance", "mentions", "entity"].edge_index = torch.tensor([ment_src, ment_dst], dtype=torch.long)
        data["utterance", "mentions", "entity"].edge_attr = torch.zeros(len(ment_src), edge_attr_dim)

    # Entity -CO_OCCURS-> Entity
    co_src, co_dst = [], []
    for r in relationships:
        if r.get("rel_type") != "CO_OCCURS":
            continue
        src_id = r.get("src_entity_id")
        dst_id = r.get("dst_entity_id")
        if src_id in entity_to_idx and dst_id in entity_to_idx:
            co_src.append(entity_to_idx[src_id])
            co_dst.append(entity_to_idx[dst_id])
    if co_src:
        data["entity", "co_occurs", "entity"].edge_index = torch.tensor([co_src, co_dst], dtype=torch.long)
        # Edge attributes: delta_t, count, recency + TGAT-style time encoding of last_seen_at
        edge_attr_list = []
        for r in relationships:
            if r.get("rel_type") != "CO_OCCURS":
                continue
            src_id, dst_id = r.get("src_entity_id"), r.get("dst_entity_id")
            if src_id in entity_to_idx and dst_id in entity_to_idx:
                dt = r.get("last_seen_at", 0) - r.get("first_seen_at", 0)
                count = r.get("count", 1)
                recency = 1.0 / (1.0 + dt / 86400)
                edge_attr_list.append([dt, float(count), recency, r.get("weight", 1.0)])
        base_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        last_ts = torch.tensor([r.get("last_seen_at", 0) for r in relationships if r.get("rel_type") == "CO_OCCURS" and r.get("src_entity_id") in entity_to_idx and r.get("dst_entity_id") in entity_to_idx], dtype=torch.float32)
        time_enc = sinusoidal_time_encoding(last_ts, time_encoding_dim)
        data["entity", "co_occurs", "entity"].edge_attr = torch.cat([base_attr, time_enc], dim=1)

    data["person"].num_nodes = data["person"].x.size(0)
    data["device"].num_nodes = data["device"].x.size(0)
    data["session"].num_nodes = data["session"].x.size(0)
    data["utterance"].num_nodes = data["utterance"].x.size(0)
    data["intent"].num_nodes = data["intent"].x.size(0)
    data["entity"].num_nodes = data["entity"].x.size(0)

    return data
