"""BM25 indexing and retrieval utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from utils.text_utils import tokenize_text

_ACTIVE_BM25_INDEX: Optional[Dict[str, object]] = None


def build_bm25_index(chunks: Sequence[Dict[str, str]]) -> Dict[str, object]:
    """Build a BM25 index bundle from chunk records."""
    if not chunks:
        raise ValueError("Cannot build BM25 index from empty chunks.")

    chunk_list = list(chunks)
    tokenized_corpus = [tokenize_text(chunk.get("text", "")) for chunk in chunk_list]
    bm25 = BM25Okapi(tokenized_corpus)

    index_bundle = {
        "chunks": chunk_list,
        "tokenized_corpus": tokenized_corpus,
        "bm25": bm25,
    }
    global _ACTIVE_BM25_INDEX
    _ACTIVE_BM25_INDEX = index_bundle
    return index_bundle


def _resolve_bundle(index_bundle: Optional[Dict[str, object]]) -> Dict[str, object]:
    if index_bundle is not None:
        return index_bundle
    if _ACTIVE_BM25_INDEX is None:
        raise ValueError("No BM25 index is active. Call build_bm25_index() first or pass index_bundle.")
    return _ACTIVE_BM25_INDEX


def get_bm25_scores(query: str, index_bundle: Optional[Dict[str, object]] = None) -> np.ndarray:
    """Return BM25 scores for all chunks for a query."""
    index_bundle = _resolve_bundle(index_bundle)
    bm25: BM25Okapi = index_bundle["bm25"]  # type: ignore[assignment]
    query_tokens = tokenize_text(query)
    return np.array(bm25.get_scores(query_tokens), dtype=np.float32)


def retrieve_bm25(
    query: str,
    k: int,
    index_bundle: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Retrieve top-k chunks using BM25."""
    index_bundle = _resolve_bundle(index_bundle)
    chunks: List[Dict[str, str]] = index_bundle["chunks"]  # type: ignore[assignment]
    if not chunks:
        return []

    scores = get_bm25_scores(query, index_bundle)
    top_k = min(k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: List[Dict[str, object]] = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[int(idx)]
        score = float(scores[int(idx)])
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "text": chunk.get("text"),
                "score": score,
                "bm25_score": score,
            }
        )
    return results


def save_bm25_index(index_bundle: Dict[str, object], output_path: Path) -> None:
    """Persist BM25 index bundle to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(index_bundle, f)


def load_bm25_index(index_path: Path) -> Dict[str, object]:
    """Load a persisted BM25 index bundle."""
    global _ACTIVE_BM25_INDEX
    with Path(index_path).open("rb") as f:
        _ACTIVE_BM25_INDEX = pickle.load(f)
    return _ACTIVE_BM25_INDEX
