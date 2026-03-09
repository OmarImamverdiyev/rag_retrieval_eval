"""Chunk Azerbaijani news articles into retrieval-friendly passages."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.text_utils import chunk_tokens, detokenize, normalize_whitespace, tokenize_text


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)


def load_news_json(path: Path) -> List[Dict[str, str]]:
    """Load news records from JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("news.json must contain a JSON list.")
    return data


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV rows using common UTF-8 variants."""
    _set_csv_field_limit()
    last_error: Optional[UnicodeDecodeError] = None
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            with Path(path).open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return []


def load_news_csv(
    path: Path,
    id_col: str = "id",
    title_col: str = "title",
    text_col: str = "content",
    link_col: str = "link",
    date_col: str = "date_time",
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load news records from CSV and map them to project schema."""
    rows = _read_csv(path)
    if not rows:
        return []

    sample_cols = set(rows[0].keys())
    required = {id_col, text_col}
    missing = sorted(column for column in required if column not in sample_cols)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    records: List[Dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        raw_doc_id = normalize_whitespace(str(row.get(id_col, "")))
        doc_id = raw_doc_id or f"csv_doc_{idx}"

        title = normalize_whitespace(str(row.get(title_col, "")))
        text = normalize_whitespace(str(row.get(text_col, "")))
        if not text and not title:
            continue
        if not title and text:
            title = text[:160]

        record: Dict[str, str] = {
            "id": doc_id,
            "title": title,
            "text": text,
        }
        if link_col in row:
            record["source_link"] = normalize_whitespace(str(row.get(link_col, "")))
        if date_col in row:
            record["source_date"] = normalize_whitespace(str(row.get(date_col, "")))

        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def load_news(
    path: Path,
    id_col: str = "id",
    title_col: str = "title",
    text_col: str = "content",
    link_col: str = "link",
    date_col: str = "date_time",
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load news from JSON or CSV based on file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_news_csv(
            path=path,
            id_col=id_col,
            title_col=title_col,
            text_col=text_col,
            link_col=link_col,
            date_col=date_col,
            limit=limit,
        )
    return load_news_json(path)


def save_chunks(chunks: Sequence[Dict[str, str]], path: Path) -> None:
    """Save chunk records to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(chunks), f, ensure_ascii=False, indent=2)


def article_to_chunks(
    article: Dict[str, str],
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Dict[str, str]]:
    """Split one article into chunks."""
    doc_id_raw = article.get("id")
    doc_id = normalize_whitespace(str(doc_id_raw)) if doc_id_raw is not None else ""
    if not doc_id:
        return []

    title = normalize_whitespace(article.get("title", ""))
    body = normalize_whitespace(article.get("text", ""))
    merged_text = normalize_whitespace(f"{title} {body}".strip())
    tokens = tokenize_text(merged_text)
    token_chunks = chunk_tokens(tokens, chunk_size=chunk_size, overlap=overlap)

    chunks: List[Dict[str, str]] = []
    for i, token_chunk in enumerate(token_chunks, start=1):
        chunk_record: Dict[str, str] = {
            "chunk_id": f"{doc_id}_chunk_{i}",
            "doc_id": doc_id,
            "text": detokenize(token_chunk),
        }
        if article.get("source_link"):
            chunk_record["source_link"] = str(article["source_link"])
        if article.get("source_date"):
            chunk_record["source_date"] = str(article["source_date"])
        chunks.append(chunk_record)
    return chunks


def chunk_news_dataset(
    news_records: Sequence[Dict[str, str]],
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Dict[str, str]]:
    """Chunk all news articles in dataset."""
    all_chunks: List[Dict[str, str]] = []
    for article in news_records:
        all_chunks.extend(article_to_chunks(article, chunk_size=chunk_size, overlap=overlap))
    return all_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk Azerbaijani news into fixed-size text passages.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "news.json", help="Path to news.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunks.json",
        help="Path to save chunked JSON",
    )
    parser.add_argument("--chunk-size", type=int, default=300, help="Target chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=50, help="Token overlap between chunks")
    parser.add_argument("--id-col", type=str, default="id", help="CSV id column name")
    parser.add_argument("--title-col", type=str, default="title", help="CSV title column name")
    parser.add_argument("--text-col", type=str, default="content", help="CSV text/content column name")
    parser.add_argument("--link-col", type=str, default="link", help="CSV link column name")
    parser.add_argument("--date-col", type=str, default="date_time", help="CSV datetime column name")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on loaded rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    news = load_news(
        args.input,
        id_col=args.id_col,
        title_col=args.title_col,
        text_col=args.text_col,
        link_col=args.link_col,
        date_col=args.date_col,
        limit=args.limit,
    )
    chunks = chunk_news_dataset(news, chunk_size=args.chunk_size, overlap=args.overlap)
    save_chunks(chunks, args.output)
    print(f"Loaded {len(news)} news articles.")
    print(f"Created {len(chunks)} chunks.")
    print(f"Saved chunks to: {args.output}")


if __name__ == "__main__":
    main()
