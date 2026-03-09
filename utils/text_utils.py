"""Utility helpers for Azerbaijani text processing and ranking post-processing."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, TypeVar

_WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
T = TypeVar("T")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim edges."""
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase words with simple Unicode-aware boundaries."""
    cleaned = normalize_whitespace(text).lower()
    return _WORD_PATTERN.findall(cleaned)


def chunk_tokens(
    tokens: Sequence[str],
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[List[str]]:
    """Split token sequence into fixed-size windows with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: List[List[str]] = []
    for start in range(0, len(tokens), step):
        window = list(tokens[start : start + chunk_size])
        if not window:
            break
        chunks.append(window)
        if start + chunk_size >= len(tokens):
            break
    return chunks


def detokenize(tokens: Sequence[str]) -> str:
    """Convert a list of tokens back to a whitespace-separated string."""
    return " ".join(tokens)


def unique_preserve_order(items: Iterable[T]) -> List[T]:
    """Deduplicate while preserving first occurrence order."""
    seen = set()
    ordered: List[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
