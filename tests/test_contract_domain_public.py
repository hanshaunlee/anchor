"""
Contract tests: domain public functions accept expected params and return expected shape (dict keys, list, etc.).
No internal code—only function names and param names / return structure.
"""
import pytest
from uuid import uuid4

from domain.ingest_service import get_household_id, get_user_role, ingest_events
from domain.explain_service import build_subgraph_from_explanation, get_similar_incidents, run_deep_dive_explainer
from domain.consent import normalize_consent_state
from domain.rule_scoring import compute_rule_score
from domain.graph_service import build_graph_from_events, normalize_events as domain_normalize_events
from domain.risk_service import list_risk_signals, get_risk_signal_detail, submit_feedback
from domain.entities.display import get_entity_display_map
from domain.rings.fingerprint import ring_fingerprint, jaccard_overlap
from domain.agents.registry import get_known_agent_names, get_agent_by_name, get_agent_by_slug, get_agents_catalog
from domain.utils.time_utils import ts_to_float, event_ts_to_float, float_to_datetime


# --- get_household_id(supabase, user_id) -> str | None ---
def test_get_household_id_signature_accepts_supabase_user_id() -> None:
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    out = get_household_id(mock, "user-1")
    assert out is None or isinstance(out, str)


def test_get_household_id_with_mock_list_returns_household_id() -> None:
    from unittest.mock import MagicMock
    mock = MagicMock()
    q = mock.table.return_value.select.return_value.eq.return_value.limit.return_value
    q.execute.return_value.data = [{"household_id": "hh-1"}]
    out = get_household_id(mock, "user-1")
    assert out == "hh-1"


# --- get_user_role(supabase, user_id) -> str | None ---
def test_get_user_role_signature() -> None:
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    out = get_user_role(mock, "user-1")
    assert out is None or isinstance(out, str)


# --- ingest_events(body, household_id, supabase) -> IngestEventsResponse ---
def test_ingest_events_accepts_body_household_supabase() -> None:
    from unittest.mock import MagicMock
    from api.schemas import IngestEventsRequest
    mock = MagicMock()
    mock.table.return_value.upsert.return_value.execute.return_value.data = []
    body = IngestEventsRequest(events=[])
    out = ingest_events(body, "hh1", mock)
    assert out is not None
    assert hasattr(out, "session_ids") or hasattr(out, "events") or isinstance(out, dict)


# --- build_subgraph_from_explanation(explanation, prefer_key=...) -> RiskSignalDetailSubgraph | None ---
@pytest.mark.parametrize("prefer_key", [None, "model_subgraph", "subgraph"])
def test_build_subgraph_from_explanation_prefer_key(prefer_key: str | None) -> None:
    out = build_subgraph_from_explanation({}, prefer_key=prefer_key)
    assert out is None or (hasattr(out, "nodes") and hasattr(out, "edges"))


def test_build_subgraph_from_explanation_with_nodes_returns_subgraph() -> None:
    out = build_subgraph_from_explanation({"model_subgraph": {"nodes": [{"id": "n1"}], "edges": []}})
    assert out is not None
    assert len(out.nodes) == 1
    assert out.nodes[0].id == "n1"


@pytest.mark.parametrize("empty_input", [{}, {"nodes": []}, {"edges": []}])
def test_build_subgraph_from_explanation_empty_returns_none(empty_input: dict) -> None:
    out = build_subgraph_from_explanation({"model_subgraph": empty_input})
    if not (empty_input.get("nodes") or empty_input.get("edges")):
        assert out is None or (len(out.nodes) == 0 and len(out.edges) == 0)


# --- get_similar_incidents(signal_id, household_id, supabase, top_k=5) -> SimilarIncidentsResponse ---
def test_get_similar_incidents_returns_response_shape() -> None:
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
    out = get_similar_incidents(uuid4(), "hh1", mock, top_k=3)
    assert hasattr(out, "available") or isinstance(out, dict) or hasattr(out, "incidents")


# --- normalize_consent_state(raw) -> dict ---
@pytest.mark.parametrize("raw", [None, {}, {"share_with_caregiver": True}])
def test_normalize_consent_state_returns_dict(raw: dict | None) -> None:
    out = normalize_consent_state(raw)
    assert isinstance(out, dict)


# --- compute_rule_score(pattern_tags, structural_motifs, entity_meta) -> float ---
@pytest.mark.parametrize("tags_len", [0, 1, 3])
def test_compute_rule_score_returns_float(tags_len: int) -> None:
    tags = ["tag"] * tags_len
    motifs = []
    meta = {}
    out = compute_rule_score(tags, motifs, meta)
    assert isinstance(out, (int, float))


# --- build_graph_from_events(household_id, events, supabase=...) -> utterances, entities, mentions, relationships ---
def test_build_graph_from_events_returns_four_lists() -> None:
    u, e, m, r = build_graph_from_events("h", [], supabase=None)
    assert isinstance(u, list)
    assert isinstance(e, list)
    assert isinstance(m, list)
    assert isinstance(r, list)


@pytest.mark.parametrize("event_count", [0, 1, 2])
def test_build_graph_from_events_with_events(event_count: int) -> None:
    events = [{"session_id": "s1", "device_id": "d1", "ts": "2025-01-01T00:00:00Z", "event_type": "wake", "payload": {}} for _ in range(event_count)]
    u, e, m, r = build_graph_from_events("h", events, supabase=None)
    assert isinstance(u, list) and isinstance(e, list)


# --- list_risk_signals(household_id, supabase, ...) -> RiskSignalListResponse ---
def test_list_risk_signals_returns_response_with_signals() -> None:
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.data = []
    mock.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.count = 0
    out = list_risk_signals("hh1", mock, max_age_days=None)
    assert hasattr(out, "signals") and isinstance(out.signals, list)


# --- get_risk_signal_detail(signal_id, household_id, supabase) -> detail | None ---
def test_get_risk_signal_detail_not_found_returns_none() -> None:
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.single.return_value.execute.return_value.data = None
    out = get_risk_signal_detail(uuid4(), "hh1", mock)
    assert out is None


# --- submit_feedback(signal_id, household_id, body, user_id, supabase) -> None, raises ValueError if not found ---
def test_submit_feedback_not_found_raises() -> None:
    from unittest.mock import MagicMock
    from api.schemas import FeedbackSubmit, FeedbackLabel
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.single.return_value.execute.return_value.data = None
    body = FeedbackSubmit(label=FeedbackLabel.true_positive, notes=None)
    with pytest.raises(ValueError, match="not found"):
        submit_feedback(uuid4(), "hh1", body, "user-1", mock)


# --- get_entity_display_map(supabase, household_id, entity_ids) -> dict ---
def test_get_entity_display_map_returns_dict() -> None:
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.table.return_value.select.return_value.in_.return_value.execute.return_value.data = []
    out = get_entity_display_map(mock, "hh1", [])
    assert isinstance(out, dict)


# --- ring_fingerprint(member_entity_ids) -> str ---
@pytest.mark.parametrize("members", [[], ["a"], ["a", "b", "c"]])
def test_ring_fingerprint_returns_str(members: list) -> None:
    out = ring_fingerprint(members)
    assert isinstance(out, str)


# --- jaccard_overlap(a, b) -> float ---
@pytest.mark.parametrize("a,b", [(set(), set()), ({"x"}, {"x"}), ({"a", "b"}, {"b", "c"})])
def test_jaccard_overlap_returns_float(a: set, b: set) -> None:
    out = jaccard_overlap(a, b)
    assert isinstance(out, (int, float))
    assert 0 <= out <= 1


# --- get_known_agent_names() -> tuple ---
def test_get_known_agent_names_returns_tuple() -> None:
    out = get_known_agent_names()
    assert isinstance(out, (tuple, list))


# --- get_agent_by_name(name) -> dict | None ---
@pytest.mark.parametrize("name", ["financial_security", "nonexistent_agent_xyz"])
def test_get_agent_by_name_returns_dict_or_none(name: str) -> None:
    out = get_agent_by_name(name)
    assert out is None or isinstance(out, dict)


# --- get_agent_by_slug(slug) -> dict | None ---
def test_get_agent_by_slug_returns_dict_or_none() -> None:
    out = get_agent_by_slug("financial")
    assert out is None or isinstance(out, dict)


# --- get_agents_catalog() -> list/dict ---
def test_get_agents_catalog_returns_iterable() -> None:
    out = get_agents_catalog()
    assert hasattr(out, "__iter__")


# --- ts_to_float(ts) ---
@pytest.mark.parametrize("ts", [None, "2025-01-01T00:00:00Z", 0, 1704067200.0])
def test_ts_to_float_accepts_ts(ts) -> None:
    out = ts_to_float(ts)
    assert isinstance(out, (int, float)) or out is None


# --- event_ts_to_float(event) ---
def test_event_ts_to_float_accepts_event_dict() -> None:
    out = event_ts_to_float({"ts": "2025-01-01T00:00:00Z"})
    assert isinstance(out, (int, float))
