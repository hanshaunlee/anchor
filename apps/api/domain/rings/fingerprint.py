"""Ring fingerprint and overlap for dedupe on persist."""
from __future__ import annotations

import hashlib
from typing import Sequence


def ring_fingerprint(member_entity_ids: Sequence[str]) -> str:
    """Strict fingerprint: sha256 of sorted member entity IDs."""
    if not member_entity_ids:
        return hashlib.sha256(b"empty").hexdigest()
    normalized = "|".join(sorted(str(x).strip() for x in member_entity_ids if x))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def jaccard_overlap(a: set[str], b: set[str]) -> float:
    """Jaccard similarity: |intersection| / |union|. Returns 0 if both empty."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
