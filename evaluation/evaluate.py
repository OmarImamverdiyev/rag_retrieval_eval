"""Retriever evaluation loop for Azerbaijani news retrieval experiments."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from evaluation.metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank
from utils.text_utils import unique_preserve_order


RetrieverFn = Callable[[str, int], List[Dict[str, object]]]


def collapse_chunk_results_to_docs(results: Sequence[Dict[str, object]]) -> List[str]:
    """Convert chunk-level results to a unique ranked list of document ids."""
    doc_ids = [str(item.get("doc_id")) for item in results if item.get("doc_id") is not None]
    return unique_preserve_order(doc_ids)


def evaluate_retriever(
    queries: Sequence[Dict[str, str]],
    retrieve_fn: RetrieverFn,
    k_values: Sequence[int] = (5, 10),
    show_progress: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate one retriever on all queries and return aggregate + per-query metrics."""
    if not queries:
        summary = {"MRR": 0.0}
        for k in sorted(set(int(x) for x in k_values)):
            summary[f"Hit@{k}"] = 0.0
            summary[f"Precision@{k}"] = 0.0
            summary[f"Recall@{k}"] = 0.0
        return summary, pd.DataFrame()

    k_values = sorted(set(int(k) for k in k_values))
    max_k = max(k_values)

    rows: List[Dict[str, object]] = []
    iterator = tqdm(queries, desc="Evaluating queries", disable=not show_progress)
    for query in iterator:
        question = query["question"]
        query_id = query["query_id"]
        relevant_doc_id = query["relevant_doc_id"]

        retrieved_chunks = retrieve_fn(question, max_k)
        # Metrics are computed at document level even though retrieval happens over chunks.
        ranked_doc_ids = collapse_chunk_results_to_docs(retrieved_chunks)

        row: Dict[str, object] = {"RR": reciprocal_rank(ranked_doc_ids, relevant_doc_id)}
        for k in k_values:
            row[f"Hit@{k}"] = hit_at_k(ranked_doc_ids, relevant_doc_id, k)
            row[f"Precision@{k}"] = precision_at_k(ranked_doc_ids, relevant_doc_id, k)
            row[f"Recall@{k}"] = recall_at_k(ranked_doc_ids, relevant_doc_id, k)
        row["query_id"] = query_id
        rows.append(row)

    query_df = pd.DataFrame(rows)
    summary: Dict[str, float] = {"MRR": float(query_df["RR"].mean())}
    for k in k_values:
        summary[f"Hit@{k}"] = float(query_df[f"Hit@{k}"].mean())
        summary[f"Precision@{k}"] = float(query_df[f"Precision@{k}"].mean())
        summary[f"Recall@{k}"] = float(query_df[f"Recall@{k}"].mean())

    return summary, query_df
