"""Tests for ml.cache.embeddings: EmbeddingCache, get_similar_sessions."""
import pytest

pytest.importorskip("torch")
import torch

from ml.cache.embeddings import EmbeddingCache, get_similar_sessions


def test_embedding_cache_put_get() -> None:
    cache = EmbeddingCache(dim=8, max_size=100)
    emb = torch.randn(8)
    cache.put("id1", emb, ts=1000.0, meta={"session_id": "s1"})
    out = cache.get("id1")
    assert out is not None
    e, ts, meta = out
    assert e.shape == (8,)
    assert ts == 1000.0
    assert meta["session_id"] == "s1"


def test_embedding_cache_get_missing() -> None:
    cache = EmbeddingCache(dim=8)
    assert cache.get("nonexistent") is None


def test_embedding_cache_put_list_padded() -> None:
    cache = EmbeddingCache(dim=8)
    cache.put("id1", [1.0, 2.0, 3.0], ts=0.0)
    e, _, _ = cache.get("id1")
    assert e.shape == (8,)


def test_embedding_cache_matrix_empty() -> None:
    cache = EmbeddingCache(dim=8)
    m = cache.matrix()
    assert m.shape == (0, 8)


def test_embedding_cache_matrix() -> None:
    cache = EmbeddingCache(dim=4)
    cache.put("a", torch.randn(4), 1.0)
    cache.put("b", torch.randn(4), 2.0)
    m = cache.matrix()
    assert m.shape == (2, 4)


def test_embedding_cache_eviction() -> None:
    cache = EmbeddingCache(dim=4, max_size=2)
    cache.put("a", torch.randn(4), 1.0)
    cache.put("b", torch.randn(4), 2.0)
    cache.put("c", torch.randn(4), 3.0)
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_embedding_cache_ids_meta_list() -> None:
    cache = EmbeddingCache(dim=4)
    cache.put("x", torch.randn(4), 0.0, meta={"k": "v"})
    assert cache.ids() == ["x"]
    assert cache.meta_list() == [{"k": "v"}]


def test_get_similar_sessions_empty() -> None:
    cache = EmbeddingCache(dim=4)
    out = get_similar_sessions(cache, torch.randn(4), top_k=3)
    assert out == []


def test_get_similar_sessions() -> None:
    cache = EmbeddingCache(dim=4)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cache.put("a", torch.tensor([1.0, 0.1, 0.0, 0.0]), 0.0)
    cache.put("b", torch.tensor([0.0, 1.0, 0.0, 0.0]), 0.0)
    out = get_similar_sessions(cache, q, top_k=2)
    assert len(out) <= 2
    assert out[0][0] == "a"
    assert out[0][1] > 0.9


def test_get_similar_sessions_exclude_id() -> None:
    cache = EmbeddingCache(dim=4)
    cache.put("a", torch.randn(4), 0.0)
    cache.put("b", torch.randn(4), 0.0)
    out = get_similar_sessions(cache, cache.matrix()[0], top_k=2, exclude_id="a")
    assert all(id_ != "a" for id_, _, _ in out)
