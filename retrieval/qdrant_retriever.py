"""Embedding indexing and retrieval utilities backed by Qdrant."""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "rag_retrieval_eval_chunks"
_ACTIVE_QDRANT_INDEX: Optional[Dict[str, object]] = None


def _normalize_l2(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def _build_client(qdrant_url: str, qdrant_api_key: Optional[str]) -> QdrantClient:
    if qdrant_api_key:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return QdrantClient(url=qdrant_url)


def _ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate_collection: bool = False,
) -> None:
    if recreate_collection and client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _qdrant_point_id(collection_name: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection_name}:{chunk_id}"))


def _upsert_points(
    client: QdrantClient,
    collection_name: str,
    points: Sequence[PointStruct],
    batch_size: int = 256,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    for start in range(0, len(points), batch_size):
        batch = list(points[start : start + batch_size])
        if not batch:
            continue
        client.upsert(collection_name=collection_name, points=batch, wait=True)


def build_qdrant_index(
    chunks: Sequence[Dict[str, str]],
    model_name: str,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    qdrant_api_key: Optional[str] = None,
    batch_size: int = 32,
    upsert_batch_size: int = 256,
    recreate_collection: bool = False,
    show_progress: bool = True,
) -> Dict[str, object]:
    """Embed chunks and persist vectors/payload to a Qdrant collection."""
    if not chunks:
        raise ValueError("Cannot build Qdrant index from empty chunks.")

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
    client = _build_client(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)
    _ensure_collection(
        client=client,
        collection_name=collection_name,
        vector_size=dim,
        recreate_collection=recreate_collection,
    )

    points: List[PointStruct] = []
    point_id_to_index: Dict[str, int] = {}
    chunk_id_to_index: Dict[str, int] = {}
    for idx, (chunk, vector) in enumerate(zip(chunk_list, embeddings)):
        chunk_id = str(chunk.get("chunk_id", f"chunk_{idx + 1}"))
        point_id = _qdrant_point_id(collection_name=collection_name, chunk_id=chunk_id)
        point_id_to_index[point_id] = idx
        chunk_id_to_index[chunk_id] = idx

        payload = {
            "doc_id": str(chunk.get("doc_id", "")),
            "chunk_id": chunk_id,
            "text": str(chunk.get("text", "")),
        }
        if chunk.get("source_link"):
            payload["source_link"] = str(chunk["source_link"])
        if chunk.get("source_date"):
            payload["source_date"] = str(chunk["source_date"])

        points.append(
            PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            )
        )

    _upsert_points(
        client=client,
        collection_name=collection_name,
        points=points,
        batch_size=upsert_batch_size,
    )

    index_bundle = {
        "chunks": chunk_list,
        "qdrant_client": client,
        "collection_name": collection_name,
        "model": model,
        "model_name": model_name,
        "point_id_to_index": point_id_to_index,
        "chunk_id_to_index": chunk_id_to_index,
    }
    global _ACTIVE_QDRANT_INDEX
    _ACTIVE_QDRANT_INDEX = index_bundle
    return index_bundle


def _resolve_bundle(index_bundle: Optional[Dict[str, object]]) -> Dict[str, object]:
    if index_bundle is not None:
        return index_bundle
    if _ACTIVE_QDRANT_INDEX is None:
        raise ValueError("No Qdrant index is active. Call build_qdrant_index() first or pass index_bundle.")
    return _ACTIVE_QDRANT_INDEX


def _encode_query(query: str, index_bundle: Optional[Dict[str, object]] = None) -> np.ndarray:
    bundle = _resolve_bundle(index_bundle)
    model: SentenceTransformer = bundle["model"]  # type: ignore[assignment]
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return _normalize_l2(vector)


def _search_qdrant(
    index_bundle: Dict[str, object],
    query_vector: np.ndarray,
    limit: int,
):
    client: QdrantClient = index_bundle["qdrant_client"]  # type: ignore[assignment]
    collection_name: str = str(index_bundle["collection_name"])
    query = query_vector[0].tolist()
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query,
            limit=limit,
            with_payload=True,
        )
        return list(response.points)
    return client.search(
        collection_name=collection_name,
        query_vector=query,
        limit=limit,
        with_payload=True,
    )


def _resolve_hit_chunk_index(hit, index_bundle: Dict[str, object]) -> Optional[int]:
    point_id_to_index: Dict[str, int] = index_bundle["point_id_to_index"]  # type: ignore[assignment]
    chunk_id_to_index: Dict[str, int] = index_bundle["chunk_id_to_index"]  # type: ignore[assignment]

    point_id = str(getattr(hit, "id", ""))
    if point_id in point_id_to_index:
        return int(point_id_to_index[point_id])

    payload = getattr(hit, "payload", {}) or {}
    payload_chunk_id = payload.get("chunk_id")
    if payload_chunk_id is not None:
        payload_chunk_id_str = str(payload_chunk_id)
        if payload_chunk_id_str in chunk_id_to_index:
            return int(chunk_id_to_index[payload_chunk_id_str])
    return None


def retrieve_qdrant(
    query: str,
    k: int,
    index_bundle: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Retrieve top-k chunks using vectors stored in Qdrant."""
    bundle = _resolve_bundle(index_bundle)
    chunks: List[Dict[str, str]] = bundle["chunks"]  # type: ignore[assignment]
    if not chunks:
        return []

    top_k = min(k, len(chunks))
    query_vector = _encode_query(query, bundle)
    hits = _search_qdrant(bundle, query_vector, top_k)

    results: List[Dict[str, object]] = []
    for rank, hit in enumerate(hits, start=1):
        idx = _resolve_hit_chunk_index(hit, bundle)
        payload = getattr(hit, "payload", {}) or {}
        chunk = chunks[idx] if idx is not None and 0 <= idx < len(chunks) else {}
        score = float(getattr(hit, "score", 0.0))
        results.append(
            {
                "rank": rank,
                "chunk_id": payload.get("chunk_id") or chunk.get("chunk_id"),
                "doc_id": payload.get("doc_id") or chunk.get("doc_id"),
                "text": payload.get("text") or chunk.get("text"),
                "score": score,
                "embedding_score": score,
            }
        )
    return results


def get_qdrant_scores(
    query: str,
    index_bundle: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """Return dense similarity scores for all chunks from Qdrant."""
    bundle = _resolve_bundle(index_bundle)
    chunks: List[Dict[str, str]] = bundle["chunks"]  # type: ignore[assignment]
    if not chunks:
        return np.array([], dtype=np.float32)

    query_vector = _encode_query(query, bundle)
    hits = _search_qdrant(bundle, query_vector, len(chunks))
    dense = np.full(len(chunks), -np.inf, dtype=np.float32)
    for hit in hits:
        idx = _resolve_hit_chunk_index(hit, bundle)
        if idx is None or idx < 0 or idx >= len(chunks):
            continue
        dense[idx] = float(getattr(hit, "score", 0.0))
    return dense
