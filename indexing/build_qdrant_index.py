"""Build and push chunk embeddings to a Qdrant collection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.embedding_retriever import DEFAULT_MODEL_NAME
from retrieval.qdrant_retriever import build_qdrant_index


def load_chunks(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("chunks.json must contain a JSON list.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qdrant index from chunked news.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunks.json",
        help="Path to chunk JSON file",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--qdrant-collection", type=str, default="rag_retrieval_eval_chunks")
    parser.add_argument("--qdrant-api-key", type=str, default=None)
    parser.add_argument("--upsert-batch-size", type=int, default=256)
    parser.add_argument("--recreate-collection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.qdrant_api_key or os.getenv("QDRANT_API_KEY")
    chunks = load_chunks(args.chunks_path)
    build_qdrant_index(
        chunks=chunks,
        model_name=args.model_name,
        qdrant_url=args.qdrant_url,
        collection_name=args.qdrant_collection,
        qdrant_api_key=api_key,
        batch_size=args.batch_size,
        upsert_batch_size=args.upsert_batch_size,
        recreate_collection=args.recreate_collection,
    )
    print(f"Loaded {len(chunks)} chunks.")
    print(f"Qdrant URL: {args.qdrant_url}")
    print(f"Collection updated: {args.qdrant_collection}")


if __name__ == "__main__":
    main()
