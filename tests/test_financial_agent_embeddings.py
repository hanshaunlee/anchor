"""Tests for Financial Security Agent: risk_signal_embeddings persist when model runs.

Phase 1 acceptance: run_financial_security_playbook with model_available=True and
embedding returned -> risk_signal_embeddings row created. When model_available=False
-> no embedding row.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from api.schemas import RiskScoreItem, RiskScoringModelMeta, RiskScoringResponse
from domain.agents.financial_security_agent import run_financial_security_playbook


def test_financial_agent_persists_embedding_when_model_returns_embedding() -> None:
    """When score_risk returns model_available=True and one score with embedding, persist path writes risk_signal_embeddings."""
    household_id = "hh-embed-1"
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "Medicare called", "confidence": 0.9, "speaker": {"role": "elder"}}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:01Z", "seq": 1, "event_type": "intent", "payload": {"name": "share_ssn", "slots": {}, "confidence": 0.85}},
    ]
    embedding_vec = [0.1] * 32
    response_with_embedding = RiskScoringResponse(
        model_available=True,
        scores=[
            RiskScoreItem(
                node_type="entity",
                node_index=0,
                score=0.6,
                signal_type="possible_scam_contact",
                embedding=embedding_vec,
                model_subgraph=None,
                model_available=True,
            ),
        ],
        model_meta=RiskScoringModelMeta(model_name="hgt_baseline", checkpoint_id="/runs/best.pt", embedding_dim=32),
    )

    tables: dict[str, MagicMock] = {}
    def table_side_effect(name):
        if name not in tables:
            t = MagicMock()
            if name == "risk_signals":
                ins = MagicMock()
                ins.execute.return_value.data = [{"id": "rs-123", "ts": "2024-01-15T10:00:00Z"}]
                t.insert.return_value = ins
            elif name == "risk_signal_embeddings":
                t.upsert.return_value = MagicMock()
            elif name == "agent_runs":
                ins = MagicMock()
                ins.execute.return_value.data = [{"id": "run-1"}]
                t.insert.return_value = ins
                t.update.return_value.eq.return_value = MagicMock()
            elif name == "watchlists":
                t.insert.return_value = MagicMock()
            else:
                t.insert.return_value = MagicMock()
            tables[name] = t
        return tables[name]
    mock_supabase = MagicMock()
    mock_supabase.table.side_effect = table_side_effect

    # So that _detect_risk_patterns gets one entity (and our mocked score_risk returns one score)
    one_utterance = [{"session_id": "s1", "text": "Medicare called", "seq": 0}]
    one_entity = [{"id": "ent-1", "entity_type": "topic", "canonical": "medicare"}]
    def fake_normalize(_household_id, _events):
        return one_utterance, one_entity, [], []

    with patch("domain.risk_scoring_service.score_risk", return_value=response_with_embedding), patch("domain.agents.financial_security_agent.normalize_events", side_effect=fake_normalize):
        result = run_financial_security_playbook(
            household_id=household_id,
            time_window_days=7,
            consent_state={"share_with_caregiver": True, "watchlist_ok": True},
            ingested_events=events,
            supabase=mock_supabase,
            dry_run=False,
            persist_score_min=0.2,  # combined = 0.4*model_score when no motifs; 0.6 -> 0.24, so need <=0.24
        )

    # Agent should produce one signal (with embedding) and persist it
    assert len(result.get("risk_signals", [])) == 1, "expected one risk signal from mocked score_risk"
    assert result["risk_signals"][0].get("embedding") == embedding_vec
    # Persist path: risk_signals.insert().execute() must return .data[0]["id"]
    assert result.get("inserted_signal_ids") == ["rs-123"], "persist should append rs-123 when insert returns id"
    emb_table_mock = tables["risk_signal_embeddings"]
    emb_table_mock.upsert.assert_called_once()
    call_kw = emb_table_mock.upsert.call_args[0][0]
    assert call_kw["risk_signal_id"] == "rs-123"
    assert call_kw["household_id"] == household_id
    assert call_kw["has_embedding"] is True
    assert call_kw["dim"] == 32
    assert call_kw["embedding"] == embedding_vec
    assert call_kw.get("model_name") is not None


def test_financial_agent_no_embedding_row_when_model_unavailable() -> None:
    """When score_risk returns model_available=False, no risk_signal_embeddings upsert."""
    household_id = "hh-no-emb"
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "Someone called", "confidence": 0.9, "speaker": {"role": "elder"}}},
    ]
    response_no_model = RiskScoringResponse(model_available=False, scores=[])

    tables: dict[str, MagicMock] = {}
    def table_side_effect(name):
        if name not in tables:
            t = MagicMock()
            if name == "risk_signals":
                ins = MagicMock()
                ins.execute.return_value.data = [{"id": "rs-456", "ts": "2024-01-15T10:00:00Z"}]
                t.insert.return_value = ins
            elif name == "risk_signal_embeddings":
                t.upsert = MagicMock()
            elif name == "agent_runs":
                ins = MagicMock()
                ins.execute.return_value.data = [{"id": "run-2"}]
                t.insert.return_value = ins
                t.update.return_value.eq.return_value = MagicMock()
            elif name == "watchlists":
                t.insert.return_value = MagicMock()
            else:
                t.insert.return_value = MagicMock()
            tables[name] = t
        return tables[name]
    mock_supabase = MagicMock()
    mock_supabase.table.side_effect = table_side_effect

    with patch("domain.risk_scoring_service.score_risk", return_value=response_no_model):
        run_financial_security_playbook(
            household_id=household_id,
            time_window_days=7,
            consent_state={"share_with_caregiver": True, "watchlist_ok": True},
            ingested_events=events,
            supabase=mock_supabase,
            dry_run=False,
            persist_score_min=0.2,
        )

    if "risk_signal_embeddings" in tables:
        tables["risk_signal_embeddings"].upsert.assert_not_called()
    # If no signals persisted (e.g. score below threshold), table might not have been created; then no upsert is correct
    assert "risk_signal_embeddings" not in tables or not tables["risk_signal_embeddings"].upsert.called
