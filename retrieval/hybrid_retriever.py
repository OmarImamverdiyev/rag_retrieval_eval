"""Hybrid retrieval combining BM25 and embedding scores."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from retrieval.bm25_retriever import get_bm25_scores
from retrieval.embedding_retriever import get_embedding_scores


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    normalized = np.zeros_like(scores)

    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return normalized

    finite_scores = scores[finite_mask]
    min_score = float(np.min(finite_scores))
    max_score = float(np.max(finite_scores))
    if max_score - min_score < 1e-12:
        return normalized

    normalized[finite_mask] = (finite_scores - min_score) / (max_score - min_score)
    return normalized


def retrieve_hybrid(
    query: str,
    k: int,
    bm25_bundle: Dict[str, object],
    embedding_bundle: Dict[str, object],
    alpha: float = 0.5,
    embedding_score_fn: Optional[Callable[[str, Dict[str, object]], np.ndarray]] = None,
) -> List[Dict[str, object]]:
    """Retrieve top-k chunks by weighted BM25 + embedding score."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    bm25_chunks = bm25_bundle["chunks"]  # type: ignore[index]
    embedding_chunks = embedding_bundle["chunks"]  # type: ignore[index]
    if len(bm25_chunks) != len(embedding_chunks):
        raise ValueError("BM25 and embedding indexes must be built from the same chunk list.")

    bm25_scores = get_bm25_scores(query, bm25_bundle)
    score_fn = embedding_score_fn or get_embedding_scores
    embedding_scores = score_fn(query, embedding_bundle)

    bm25_norm = _min_max_normalize(bm25_scores)
    embedding_norm = _min_max_normalize(embedding_scores)
    hybrid_scores = alpha * bm25_norm + (1.0 - alpha) * embedding_norm

    top_k = min(k, len(bm25_chunks))
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results: List[Dict[str, object]] = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = bm25_chunks[int(idx)]
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "text": chunk.get("text"),
                "score": float(hybrid_scores[int(idx)]),
                "hybrid_score": float(hybrid_scores[int(idx)]),
                "bm25_score_normalized": float(bm25_norm[int(idx)]),
                "embedding_score_normalized": float(embedding_norm[int(idx)]),
            }
        )
    return results
