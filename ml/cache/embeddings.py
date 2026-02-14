"""
Embedding cache: (node_id/session_id, embedding, ts) for retrieval and explainers.
Stores in memory and optionally syncs to session_embeddings (Supabase/pgvector).
"""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    In-memory cache: (id, embedding tensor, ts).
    Optional: persist to session_embeddings table (embedding as JSONB or pgvector).
    """

    def __init__(self, dim: int, max_size: int = 10_000, device: str = "cpu"):
        self.dim = dim
        self.max_size = max_size
        self.device = device
        self._ids: list[str] = []
        self._embeddings: list[torch.Tensor] = []
        self._ts: list[float] = []
        self._meta: list[dict] = []  # e.g. {"session_id", "risk_signal_id", "outcome"}

    def put(self, id_: str, embedding: torch.Tensor, ts: float, meta: dict | None = None) -> None:
        if isinstance(embedding, torch.Tensor):
            emb = embedding.detach().to(self.device).float()
        else:
            emb = torch.tensor(embedding, dtype=torch.float32, device=self.device)
        if emb.dim() > 1:
            emb = emb.flatten()
        if emb.size(0) != self.dim:
            emb = emb[: self.dim] if emb.size(0) >= self.dim else torch.nn.functional.pad(emb, (0, self.dim - emb.size(0)))
        self._ids.append(id_)
        self._embeddings.append(emb)
        self._ts.append(ts)
        self._meta.append(meta or {})
        if len(self._ids) > self.max_size:
            self._ids.pop(0)
            self._embeddings.pop(0)
            self._ts.pop(0)
            self._meta.pop(0)

    def get(self, id_: str) -> tuple[torch.Tensor, float, dict] | None:
        try:
            i = self._ids.index(id_)
            return self._embeddings[i], self._ts[i], self._meta[i]
        except ValueError:
            return None

    def matrix(self) -> torch.Tensor:
        """(N, dim) for similarity search."""
        if not self._embeddings:
            return torch.empty(0, self.dim, device=self.device)
        return torch.stack(self._embeddings)

    def ids(self) -> list[str]:
        return self._ids.copy()

    def meta_list(self) -> list[dict]:
        return self._meta.copy()


def get_similar_sessions(
    cache: EmbeddingCache,
    query_embedding: torch.Tensor,
    top_k: int = 5,
    exclude_id: str | None = None,
) -> list[tuple[str, float, dict]]:
    """Cosine nearest neighbors. Returns [(id, score, meta), ...]."""
    mat = cache.matrix()
    if mat.size(0) == 0:
        return []
    q = query_embedding.flatten().to(cache.device).float()
    if q.size(0) != cache.dim:
        q = q[: cache.dim] if q.size(0) >= cache.dim else torch.nn.functional.pad(q, (0, cache.dim - q.size(0)))
    q = q / (q.norm() + 1e-8)
    mat = mat / (mat.norm(dim=1, keepdim=True) + 1e-8)
    scores = mat @ q
    idx = scores.argsort(descending=True)
    out = []
    for i in idx.tolist():
        if exclude_id and cache._ids[i] == exclude_id:
            continue
        out.append((cache._ids[i], float(scores[i]), cache._meta[i]))
        if len(out) >= top_k:
            break
    return out
