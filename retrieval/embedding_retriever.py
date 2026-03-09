"""Embedding indexing and retrieval utilities based on sentence-transformers + FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_ACTIVE_EMBEDDING_INDEX: Optional[Dict[str, object]] = None


def _normalize_l2(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def build_embedding_index(
    chunks: Sequence[Dict[str, str]],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    show_progress: bool = True,
) -> Dict[str, object]:
    """Build embedding vectors and FAISS cosine-similarity index for chunks."""
    if not chunks:
        raise ValueError("Cannot build embedding index from empty chunks.")

    chunk_list = list(chunks)
    texts = [chunk.get("text", "") for chunk in chunk_list]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = _normalize_l2(embeddings)

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_bundle = {
        "chunks": chunk_list,
        "embeddings": embeddings,
        "faiss_index": index,
        "model": model,
        "model_name": model_name,
    }
    global _ACTIVE_EMBEDDING_INDEX
    _ACTIVE_EMBEDDING_INDEX = index_bundle
    return index_bundle


def _resolve_bundle(index_bundle: Optional[Dict[str, object]]) -> Dict[str, object]:
    if index_bundle is not None:
        return index_bundle
    if _ACTIVE_EMBEDDING_INDEX is None:
        raise ValueError("No embedding index is active. Call build_embedding_index() first or pass index_bundle.")
    return _ACTIVE_EMBEDDING_INDEX


def _encode_query(query: str, index_bundle: Optional[Dict[str, object]]) -> np.ndarray:
    index_bundle = _resolve_bundle(index_bundle)
    model: SentenceTransformer = index_bundle["model"]  # type: ignore[assignment]
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    vector = _normalize_l2(vector)
    return vector


def retrieve_embedding(
    query: str,
    k: int,
    index_bundle: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Retrieve top-k chunks using embedding cosine similarity."""
    index_bundle = _resolve_bundle(index_bundle)
    chunks: List[Dict[str, str]] = index_bundle["chunks"]  # type: ignore[assignment]
    index: faiss.Index = index_bundle["faiss_index"]  # type: ignore[assignment]
    if not chunks:
        return []

    query_vector = _encode_query(query, index_bundle)
    top_k = min(k, len(chunks))
    scores, ids = index.search(query_vector, top_k)

    results: List[Dict[str, object]] = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "text": chunk.get("text"),
                "score": float(score),
                "embedding_score": float(score),
            }
        )
    return results


def get_embedding_scores(query: str, index_bundle: Optional[Dict[str, object]] = None) -> np.ndarray:
    """Return embedding similarity scores for all chunks for a query."""
    index_bundle = _resolve_bundle(index_bundle)
    chunks: List[Dict[str, str]] = index_bundle["chunks"]  # type: ignore[assignment]
    index: faiss.Index = index_bundle["faiss_index"]  # type: ignore[assignment]
    if not chunks:
        return np.array([], dtype=np.float32)

    query_vector = _encode_query(query, index_bundle)
    n_docs = len(chunks)
    scores, ids = index.search(query_vector, n_docs)

    dense_scores = np.full(n_docs, -np.inf, dtype=np.float32)
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            dense_scores[int(idx)] = float(score)
    return dense_scores


def save_embedding_index(index_bundle: Dict[str, object], output_dir: Path) -> None:
    """Save FAISS index, embeddings, and metadata needed to reload index bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index_bundle["faiss_index"], str(output_dir / "faiss.index"))  # type: ignore[arg-type]
    np.save(output_dir / "embeddings.npy", index_bundle["embeddings"])  # type: ignore[arg-type]

    metadata = {
        "model_name": index_bundle["model_name"],
        "chunks": index_bundle["chunks"],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_embedding_index(output_dir: Path) -> Dict[str, object]:
    """Load FAISS index artifacts and instantiate the embedding model."""
    output_dir = Path(output_dir)

    with (output_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_name = metadata["model_name"]
    chunks = metadata["chunks"]
    embeddings = np.load(output_dir / "embeddings.npy")
    index = faiss.read_index(str(output_dir / "faiss.index"))
    model = SentenceTransformer(model_name)

    index_bundle = {
        "chunks": chunks,
        "embeddings": embeddings,
        "faiss_index": index,
        "model": model,
        "model_name": model_name,
    }
    global _ACTIVE_EMBEDDING_INDEX
    _ACTIVE_EMBEDDING_INDEX = index_bundle
    return index_bundle
