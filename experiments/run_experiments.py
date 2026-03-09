"""Run retrieval experiments for BM25, embedding, and hybrid retrievers."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import evaluate_retriever
from indexing.chunk_news import chunk_news_dataset, load_news, save_chunks
from retrieval.bm25_retriever import build_bm25_index, retrieve_bm25
from retrieval.embedding_retriever import DEFAULT_MODEL_NAME, build_embedding_index, retrieve_embedding
from retrieval.hybrid_retriever import retrieve_hybrid
from utils.text_utils import normalize_whitespace

DEFAULT_CORPORA_APA_PATH = PROJECT_ROOT.parent / "Corpora" / "apa.az.csv"
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "queries.json"


def load_queries(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("queries.json must contain a JSON list.")
    return data


def parse_k_values(raw: str) -> List[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("k-values must contain at least one integer.")
    if any(value <= 0 for value in values):
        raise ValueError("k-values must be positive integers.")
    return sorted(set(values))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_auto_queries(news: Sequence[Dict[str, str]], max_queries: int | None = None) -> List[Dict[str, str]]:
    """Create weak labels from source docs (title -> relevant doc id)."""
    queries: List[Dict[str, str]] = []
    for article in news:
        doc_id = normalize_whitespace(str(article.get("id", "")))
        title = normalize_whitespace(str(article.get("title", "")))
        text = normalize_whitespace(str(article.get("text", "")))
        if not doc_id:
            continue
        question = title or text[:220]
        if not question:
            continue
        queries.append(
            {
                "query_id": f"auto_q_{len(queries) + 1}",
                "question": question,
                "relevant_doc_id": doc_id,
            }
        )
        if max_queries is not None and len(queries) >= max_queries:
            break

    if not queries:
        raise ValueError("Failed to auto-generate queries: no usable title/text in source news.")
    return queries


def _missing_query_labels(queries: Sequence[Dict[str, str]], news: Sequence[Dict[str, str]]) -> List[str]:
    news_ids = {normalize_whitespace(str(item.get("id", ""))) for item in news}
    missing_ids: List[str] = []
    for query in queries:
        relevant_doc_id = normalize_whitespace(str(query.get("relevant_doc_id", "")))
        if relevant_doc_id and relevant_doc_id in news_ids:
            continue
        missing_ids.append(str(query.get("query_id", "unknown_query")))
    return missing_ids


def build_queries(args: argparse.Namespace, news: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    """Load query labels or create auto labels when needed."""
    if args.auto_generate_queries:
        queries = generate_auto_queries(news, max_queries=args.max_auto_queries)
        return queries, "auto-generated from source titles/text"

    queries = load_queries(args.queries_path)
    missing_query_ids = _missing_query_labels(queries=queries, news=news)
    if not missing_query_ids:
        return queries, f"loaded from {args.queries_path}"

    default_queries = args.queries_path.resolve() == DEFAULT_QUERIES_PATH.resolve()
    csv_source = args.news_path.suffix.lower() == ".csv"
    if csv_source and default_queries:
        queries = generate_auto_queries(news, max_queries=args.max_auto_queries)
        print(
            "Default queries.json does not match CSV doc ids; "
            "falling back to auto-generated query labels."
        )
        return queries, "auto-generated from source titles/text"

    missing_preview = ", ".join(missing_query_ids[:5])
    raise ValueError(
        "Query labels do not match source document ids. "
        f"First missing query ids: {missing_preview}. "
        "Provide matching queries or run with --auto-generate-queries."
    )


def print_ascii_visualization(results_df: pd.DataFrame, metric_names: Sequence[str]) -> None:
    """Render simple terminal bars for quick visual comparison."""
    bar_width = 28
    print("\nRetrieval Performance Visualization")
    for metric in metric_names:
        if metric not in results_df.columns:
            continue
        print(f"\n{metric}")
        for _, row in results_df.iterrows():
            value = float(row[metric])
            filled = max(0, min(bar_width, int(round(value * bar_width))))
            bar = "#" * filled + "-" * (bar_width - filled)
            print(f"{row['Method']:<10} [{bar}] {value:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods for Azerbaijani news RAG.")
    parser.add_argument("--news-path", type=Path, default=PROJECT_ROOT / "data" / "news.json")
    parser.add_argument("--queries-path", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--chunks-path", type=Path, default=PROJECT_ROOT / "data" / "chunks.json")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--use-corpora-apa",
        action="store_true",
        help=f"Use default corpus CSV at {DEFAULT_CORPORA_APA_PATH}",
    )
    parser.add_argument("--source-id-col", type=str, default="id", help="CSV id column name")
    parser.add_argument("--source-title-col", type=str, default="title", help="CSV title column name")
    parser.add_argument("--source-text-col", type=str, default="content", help="CSV text/content column name")
    parser.add_argument("--source-link-col", type=str, default="link", help="CSV link column name")
    parser.add_argument("--source-date-col", type=str, default="date_time", help="CSV date/time column name")
    parser.add_argument("--news-limit", type=int, default=None, help="Optional cap on loaded source rows")
    parser.add_argument(
        "--auto-generate-queries",
        action="store_true",
        help="Generate weak labels from source docs (query=title/text, relevant_doc_id=id)",
    )
    parser.add_argument("--max-auto-queries", type=int, default=5000, help="Cap for auto-generated query set")
    parser.add_argument("--chunk-size", type=int, default=300, help="Target chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Token overlap between chunks")
    parser.add_argument("--k-values", type=str, default="5,10", help="Comma-separated K values (example: 5,10)")
    parser.add_argument("--hybrid-alpha", type=float, default=0.5, help="Hybrid weight for BM25 score")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--embedding-backend",
        choices=["faiss", "qdrant"],
        default="faiss",
        help="Where dense embeddings are indexed/searched",
    )
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument(
        "--qdrant-collection",
        type=str,
        default="rag_retrieval_eval_chunks",
        help="Target Qdrant collection for chunk vectors",
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=None,
        help="Qdrant API key (or set env QDRANT_API_KEY)",
    )
    parser.add_argument(
        "--qdrant-recreate-collection",
        action="store_true",
        help="Drop and recreate Qdrant collection before upsert",
    )
    parser.add_argument("--qdrant-upsert-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true", help="Print optional ASCII visualization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_corpora_apa:
        args.news_path = DEFAULT_CORPORA_APA_PATH
    if args.qdrant_api_key is None:
        args.qdrant_api_key = os.getenv("QDRANT_API_KEY")

    set_seed(args.seed)
    k_values = parse_k_values(args.k_values)
    max_k = max(k_values)

    news = load_news(
        path=args.news_path,
        id_col=args.source_id_col,
        title_col=args.source_title_col,
        text_col=args.source_text_col,
        link_col=args.source_link_col,
        date_col=args.source_date_col,
        limit=args.news_limit,
    )
    if not news:
        raise ValueError(f"No source documents loaded from: {args.news_path}")
    queries, query_source_note = build_queries(args=args, news=news)

    chunks = chunk_news_dataset(news, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    if not chunks:
        raise ValueError("No chunks were generated from source documents.")
    save_chunks(chunks, args.chunks_path)

    bm25_bundle = build_bm25_index(chunks)
    embedding_method_name = "Embedding(FAISS)"
    embedding_score_fn = None

    if args.embedding_backend == "faiss":
        embedding_bundle = build_embedding_index(
            chunks=chunks,
            model_name=args.model_name,
            batch_size=args.batch_size,
            show_progress=True,
        )
        embedding_retrieve_fn = lambda question, k: retrieve_embedding(question, k, embedding_bundle)
    else:
        try:
            from retrieval.qdrant_retriever import (
                build_qdrant_index,
                get_qdrant_scores,
                retrieve_qdrant,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qdrant backend requested but qdrant client dependencies are not usable. "
                "Install/upgrade dependencies with: pip install -r requirements.txt "
                "and if needed: pip install -U qdrant-client protobuf"
            ) from exc

        try:
            embedding_bundle = build_qdrant_index(
                chunks=chunks,
                model_name=args.model_name,
                qdrant_url=args.qdrant_url,
                collection_name=args.qdrant_collection,
                qdrant_api_key=args.qdrant_api_key,
                batch_size=args.batch_size,
                upsert_batch_size=args.qdrant_upsert_batch_size,
                recreate_collection=args.qdrant_recreate_collection,
                show_progress=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to build Qdrant embedding index. "
                f"Ensure Qdrant is running and reachable at {args.qdrant_url} "
                f"and the collection '{args.qdrant_collection}' is writable."
            ) from exc
        embedding_retrieve_fn = lambda question, k: retrieve_qdrant(question, k, embedding_bundle)
        embedding_score_fn = get_qdrant_scores
        embedding_method_name = "Embedding(Qdrant)"

    methods = [
        ("BM25", lambda question, k: retrieve_bm25(question, k, bm25_bundle)),
        (embedding_method_name, embedding_retrieve_fn),
        (
            "Hybrid",
            lambda question, k: retrieve_hybrid(
                question,
                k,
                bm25_bundle=bm25_bundle,
                embedding_bundle=embedding_bundle,
                alpha=args.hybrid_alpha,
                embedding_score_fn=embedding_score_fn,
            ),
        ),
    ]

    summary_rows: List[Dict[str, object]] = []
    query_rows: List[pd.DataFrame] = []
    for method_name, retrieve_fn in methods:
        summary, per_query_df = evaluate_retriever(
            queries=queries,
            retrieve_fn=retrieve_fn,
            k_values=k_values,
            show_progress=True,
        )
        summary_rows.append({"Method": method_name, **summary})
        per_query_df.insert(0, "Method", method_name)
        query_rows.append(per_query_df)

    summary_df = pd.DataFrame(summary_rows)
    ordered_columns = (
        ["Method"]
        + [f"Hit@{k}" for k in k_values]
        + [f"Recall@{k}" for k in k_values]
        + [f"Precision@{k}" for k in k_values]
        + ["MRR"]
    )
    summary_df = summary_df[[column for column in ordered_columns if column in summary_df.columns]]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "retrieval_results.csv"
    query_path = results_dir / "query_level_results.csv"
    summary_df.to_csv(summary_path, index=False)
    pd.concat(query_rows, ignore_index=True).to_csv(query_path, index=False)

    float_formatter = lambda value: f"{value:.4f}"
    print("\nRetrieval Results")
    print(summary_df.to_string(index=False, float_format=float_formatter))
    print(f"\nNews source: {args.news_path}")
    print(f"Loaded docs: {len(news)} | Generated chunks: {len(chunks)}")
    print(f"Query labels: {len(queries)} ({query_source_note})")
    print(f"Embedding backend: {args.embedding_backend}")
    if args.embedding_backend == "qdrant":
        print(f"Qdrant collection: {args.qdrant_collection} @ {args.qdrant_url}")
    print(f"\nSaved summary CSV: {summary_path}")
    print(f"Saved query-level CSV: {query_path}")
    print(f"Chunk file updated: {args.chunks_path}")
    print(f"Evaluated top-k values: {k_values} (max_k={max_k})")

    if args.visualize:
        print_ascii_visualization(summary_df, metric_names=[f"Hit@{k}" for k in k_values] + ["MRR"])


if __name__ == "__main__":
    main()
