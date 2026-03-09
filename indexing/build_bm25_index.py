"""Build and persist BM25 index artifacts from chunked news data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.bm25_retriever import build_bm25_index, save_bm25_index


def load_chunks(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("chunks.json must contain a JSON list.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BM25 index from chunked news.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunks.json",
        help="Path to chunk JSON file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "bm25_index.pkl",
        help="Path to save BM25 index pickle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_chunks(args.chunks_path)
    index_bundle = build_bm25_index(chunks)
    save_bm25_index(index_bundle, args.output_path)
    print(f"Loaded {len(chunks)} chunks.")
    print(f"Saved BM25 index to: {args.output_path}")


if __name__ == "__main__":
    main()
