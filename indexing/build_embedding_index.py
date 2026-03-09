"""Build and persist embedding + FAISS index artifacts from chunked news data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.embedding_retriever import (
    DEFAULT_MODEL_NAME,
    build_embedding_index,
    save_embedding_index,
)


def load_chunks(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("chunks.json must contain a JSON list.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build embedding index from chunked news.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunks.json",
        help="Path to chunk JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "embedding",
        help="Directory to save FAISS and metadata artifacts",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Sentence-transformers model name",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_chunks(args.chunks_path)
    index_bundle = build_embedding_index(
        chunks=chunks,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    save_embedding_index(index_bundle, args.output_dir)
    print(f"Loaded {len(chunks)} chunks.")
    print(f"Saved embedding index artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
