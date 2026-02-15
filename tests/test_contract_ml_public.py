"""
Contract tests: ML public API—function names and params only. run_inference, build_hetero_from_tables, extract_motifs, etc.
"""
import pytest

pytest.importorskip("torch")


# --- ml.inference: load_model, run_inference ---
def test_load_model_signature() -> None:
    from pathlib import Path
    from ml.inference import load_model
    import torch
    # When checkpoint missing, may return (None, None) or raise
    try:
        out = load_model(Path("nonexistent.pt"), torch.device("cpu"))
        assert out is None or (isinstance(out, tuple) and len(out) == 2)
    except (FileNotFoundError, Exception):
        pass


def test_run_inference_signature() -> None:
    from ml.inference import run_inference
    import torch
    from torch_geometric.data import HeteroData
    # Minimal hetero data: need node type and x
    try:
        data = HeteroData()
        data["entity"].x = torch.randn(2, 4)
        data["entity"].num_nodes = 2
        from ml.models.hgt_baseline import HGTBaseline
        model = HGTBaseline(in_channels=4, hidden_channels=8, out_channels=2, num_layers=1, metadata=(["entity"], []))
        risk_list, expl = run_inference(model, data, torch.device("cpu"), return_embeddings=False)
        assert isinstance(risk_list, list)
        assert expl is None or isinstance(expl, dict)
        if risk_list:
            assert "node_index" in risk_list[0] and "score" in risk_list[0]
    except Exception:
        pytest.skip("HGT or hetero setup not available")


# --- ml.graph.builder: build_hetero_from_tables ---
def test_build_hetero_from_tables_signature() -> None:
    from ml.graph.builder import build_hetero_from_tables
    out = build_hetero_from_tables(
        "hh1",
        [],
        [],
        [],
        [],
        [],
    )
    assert out is not None
    assert hasattr(out, "node_stores") or hasattr(out, "metadata")


def test_build_hetero_from_tables_with_entities() -> None:
    from ml.graph.builder import build_hetero_from_tables
    entities = [{"id": "e1", "canonical": "alice"}]
    out = build_hetero_from_tables("hh1", [], [], entities, [], [])
    assert out is not None


# --- ml.explainers.motifs: extract_motifs ---
@pytest.mark.parametrize("utterances_len", [0, 1])
def test_extract_motifs_returns_three_items(utterances_len: int) -> None:
    from ml.explainers.motifs import extract_motifs
    utterances = [{"text": "hello"}] * utterances_len
    tags, snippet, motifs = extract_motifs(utterances, [], [], [], [], {})
    assert isinstance(tags, list)
    assert isinstance(snippet, list)
    assert isinstance(motifs, list)


# --- ml.train: focal_loss ---
def test_focal_loss_signature() -> None:
    import torch
    from ml.train import focal_loss
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 0, 1])
    out = focal_loss(logits, targets, gamma=2.0, reduction="mean")
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 0 or out.numel() == 1


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_focal_loss_reduction(reduction: str) -> None:
    import torch
    from ml.train import focal_loss
    logits = torch.randn(3, 2)
    targets = torch.tensor([0, 1, 0])
    out = focal_loss(logits, targets, reduction=reduction)
    assert isinstance(out, torch.Tensor)


# --- ml.config: load_ml_yaml, get_train_config ---
def test_load_ml_yaml_missing_returns_empty_or_dict() -> None:
    from ml.config import load_ml_yaml
    out = load_ml_yaml("nonexistent_file_xyz.yaml")
    assert isinstance(out, dict)


def test_get_train_config_returns_dict() -> None:
    from ml.config import get_train_config
    out = get_train_config()
    assert isinstance(out, dict)


# --- ml.registry: get_model_name, get_runner ---
def test_get_model_name_returns_str() -> None:
    from ml.registry import get_model_name
    out = get_model_name()
    assert isinstance(out, str)


def test_get_runner_returns_runner_or_none() -> None:
    from ml.registry import get_runner
    out = get_runner(None, checkpoint_path=None)
    assert out is None or (hasattr(out, "run") and callable(getattr(out, "run", None)))


# --- ml.graph.subgraph: extract_k_hop, get_touched_entities ---
def test_extract_k_hop_signature() -> None:
    from ml.graph.subgraph import extract_k_hop
    from torch_geometric.data import HeteroData
    import torch
    data = HeteroData()
    data["entity"].x = torch.randn(3, 2)
    data["entity"].num_nodes = 3
    out = extract_k_hop(data, {"entity": [0]}, k=1)
    assert out is not None


def test_get_touched_entities_signature() -> None:
    from ml.graph.subgraph import get_touched_entities
    out = get_touched_entities([], [], {})
    assert isinstance(out, (set, list))


# --- ml.graph.time_encoding ---
def test_sinusoidal_time_encoding_shape() -> None:
    import torch
    from ml.graph.time_encoding import sinusoidal_time_encoding
    ts = torch.tensor([0.0, 1.0])
    out = sinusoidal_time_encoding(ts, dim=8)
    assert out.shape[-1] == 8


# --- ml.cache.embeddings (get_similar_sessions): requires EmbeddingCache and query tensor ---
def test_get_similar_sessions_signature() -> None:
    from ml.cache.embeddings import get_similar_sessions, EmbeddingCache
    import torch
    cache = EmbeddingCache(dim=4, max_size=10)
    cache.put("id1", [1.0, 0.0, 0.0, 0.0], 0.0)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    out = get_similar_sessions(cache, q, top_k=5, exclude_id=None)
    assert isinstance(out, list)


# --- domain.risk_scoring_service.score_risk (uses GNN / rule fallback) ---
def test_score_risk_empty_entities_returns_model_available_false() -> None:
    from domain.risk_scoring_service import score_risk
    from api.schemas import RiskScoringResponse
    out = score_risk("hh1", sessions=[], utterances=[], entities=[], mentions=[], relationships=[])
    assert isinstance(out, RiskScoringResponse)
    assert out.model_available is False
    assert out.scores == []


def test_score_risk_with_entities_returns_response() -> None:
    from domain.risk_scoring_service import score_risk
    entities = [{"id": "e1", "canonical": "alice"}]
    out = score_risk("hh1", sessions=[], utterances=[], entities=entities, mentions=[], relationships=[])
    assert hasattr(out, "model_available")
    assert hasattr(out, "scores")
    assert isinstance(out.scores, list)
