"""Information retrieval metrics used for retrieval-side RAG evaluation."""

from __future__ import annotations

from typing import Iterable, Sequence, Set, Union

RelevantType = Union[str, Sequence[str], Set[str]]


def _to_relevant_set(relevant_doc_ids: RelevantType) -> Set[str]:
    if isinstance(relevant_doc_ids, str):
        return {relevant_doc_ids}
    return set(relevant_doc_ids)


def hit_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: RelevantType, k: int) -> float:
    """Hit@K: 1 if any relevant doc appears in top-k, else 0."""
    relevant = _to_relevant_set(relevant_doc_ids)
    if k <= 0:
        return 0.0
    return float(any(doc_id in relevant for doc_id in ranked_doc_ids[:k]))


def precision_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: RelevantType, k: int) -> float:
    """Precision@K: fraction of top-k retrieved documents that are relevant."""
    if k <= 0:
        return 0.0
    relevant = _to_relevant_set(relevant_doc_ids)
    if not relevant:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / float(k)


def recall_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: RelevantType, k: int) -> float:
    """Recall@K: fraction of relevant documents found in top-k results."""
    relevant = _to_relevant_set(relevant_doc_ids)
    if not relevant or k <= 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / float(len(relevant))


def reciprocal_rank(ranked_doc_ids: Sequence[str], relevant_doc_ids: RelevantType) -> float:
    """Reciprocal rank of first relevant result."""
    relevant = _to_relevant_set(relevant_doc_ids)
    for idx, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / float(idx)
    return 0.0


def mean_reciprocal_rank(
    ranked_lists: Iterable[Sequence[str]],
    relevant_lists: Iterable[RelevantType],
) -> float:
    """Mean Reciprocal Rank across many queries."""
    scores = [reciprocal_rank(ranked, relevant) for ranked, relevant in zip(ranked_lists, relevant_lists)]
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))
