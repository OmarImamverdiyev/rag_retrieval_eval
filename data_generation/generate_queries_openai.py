"""Generate retrieval queries with OpenAI and save in project query-label format."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indexing.chunk_news import load_news
from utils.text_utils import normalize_whitespace

DEFAULT_CORPORA_APA_PATH = PROJECT_ROOT.parent / "Corpora" / "apa.az.csv"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "query_generation_prompt.txt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "queries.json"
DEFAULT_NEWS_PATH = PROJECT_ROOT / "data" / "news.json"
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_SELECTED_DOCS_PATH = PROJECT_ROOT / "data" / "selected_docs_for_queries.json"
DEFAULT_SELECTED_DOC_IDS_PATH = PROJECT_ROOT / "data" / "selected_doc_ids_for_queries.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ~N retrieval queries from news documents.")
    parser.add_argument("--news-path", type=Path, default=DEFAULT_NEWS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--target-queries", type=int, default=5000)
    parser.add_argument("--queries-per-doc", type=int, default=4)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-passage-chars", type=int, default=1400)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--min-text-words", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-between-calls", type=float, default=0.0)
    parser.add_argument(
        "--save-every-docs",
        type=int,
        default=25,
        help="Checkpoint-save generated queries every N processed docs.",
    )
    parser.add_argument(
        "--input-rate-per-1m",
        type=float,
        default=0.25,
        help="USD per 1M input tokens (standard tier).",
    )
    parser.add_argument(
        "--output-rate-per-1m",
        type=float,
        default=2.00,
        help="USD per 1M output tokens (standard tier).",
    )
    parser.add_argument(
        "--use-corpora-apa",
        action="store_true",
        help=f"Use default corpus CSV at {DEFAULT_CORPORA_APA_PATH}",
    )
    parser.add_argument("--source-id-col", type=str, default="id")
    parser.add_argument("--source-title-col", type=str, default="title")
    parser.add_argument("--source-text-col", type=str, default="content")
    parser.add_argument("--source-link-col", type=str, default="link")
    parser.add_argument("--source-date-col", type=str, default="date_time")
    parser.add_argument("--news-limit", type=int, default=None)
    parser.add_argument("--dotenv-path", type=Path, default=DEFAULT_DOTENV_PATH)
    parser.add_argument("--selected-docs-path", type=Path, default=DEFAULT_SELECTED_DOCS_PATH)
    parser.add_argument("--selected-doc-ids-path", type=Path, default=DEFAULT_SELECTED_DOC_IDS_PATH)
    parser.add_argument(
        "--reuse-selected-docs",
        action="store_true",
        help="Load selected docs from --selected-docs-path instead of re-selecting from source.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Only perform doc selection and save selected docs files.",
    )
    return parser.parse_args()


def load_prompt(path: Path) -> str:
    with Path(path).open("r", encoding="utf-8") as f:
        return f.read()


def load_json_list(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in: {path}")
    return data


def load_dotenv(path: Path) -> None:
    """Load .env key-value pairs into process env if absent."""
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if not key:
                continue
            if key not in os.environ and value:
                os.environ[key] = value


def sanitize_passage(title: str, text: str, max_chars: int) -> str:
    merged = normalize_whitespace(f"{title}\n\n{text}".strip())
    if max_chars <= 0:
        return merged
    return merged[:max_chars]


def count_words(text: str) -> int:
    parts = [p for p in normalize_whitespace(text).split(" ") if p]
    return len(parts)


def select_docs(
    news: Sequence[Dict[str, str]],
    target_queries: int,
    queries_per_doc: int,
    min_text_words: int,
    seed: int,
) -> List[Dict[str, str]]:
    if queries_per_doc <= 0:
        raise ValueError("queries_per_doc must be positive.")
    filtered = []
    for item in news:
        doc_id = normalize_whitespace(str(item.get("id", "")))
        text = normalize_whitespace(str(item.get("text", "")))
        title = normalize_whitespace(str(item.get("title", "")))
        if not doc_id:
            continue
        if not text and not title:
            continue
        if count_words(text or title) < min_text_words:
            continue
        filtered.append(item)

    if not filtered:
        raise ValueError("No eligible documents found after filtering.")

    needed_docs = math.ceil(target_queries / queries_per_doc)
    rng = random.Random(seed)
    rng.shuffle(filtered)

    if len(filtered) >= needed_docs:
        return filtered[:needed_docs]

    # If source set is smaller than required, cycle through docs.
    selected: List[Dict[str, str]] = []
    while len(selected) < needed_docs:
        block = list(filtered)
        rng.shuffle(block)
        selected.extend(block)
    return selected[:needed_docs]


def extract_queries_from_payload(payload: object) -> List[str]:
    if isinstance(payload, str):
        parsed = json.loads(payload)
    else:
        parsed = payload

    if isinstance(parsed, dict):
        items = parsed.get("queries", [])
    elif isinstance(parsed, list):
        # Backward compatibility in case model returns a raw list.
        items = parsed
    else:
        raise ValueError("Model response is not valid JSON query container.")

    if not isinstance(items, list):
        raise ValueError("Model response field 'queries' is not a list.")

    cleaned: List[str] = []
    for item in items:
        value = normalize_whitespace(str(item))
        if value:
            cleaned.append(value)
    return cleaned


def extract_queries_from_response(response: object) -> List[str]:
    candidates: List[object] = []

    output_parsed = getattr(response, "output_parsed", None)
    if output_parsed is not None:
        candidates.append(output_parsed)

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        candidates.append(output_text)

    output_items = getattr(response, "output", None)
    if isinstance(output_items, list):
        for item in output_items:
            content_items = getattr(item, "content", None)
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                parsed_value = getattr(content, "parsed", None)
                if parsed_value is not None:
                    candidates.append(parsed_value)
                json_value = getattr(content, "json", None)
                if json_value is not None:
                    candidates.append(json_value)
                text_value = getattr(content, "text", None)
                if isinstance(text_value, str) and text_value.strip():
                    candidates.append(text_value)

    errors: List[str] = []
    for candidate in candidates:
        try:
            return extract_queries_from_payload(candidate)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            continue
    raise ValueError(f"No parseable JSON query payload found. Attempts: {errors[:3]}")


def validate_queries(items: Sequence[str], expected_count: int) -> Tuple[bool, str]:
    if len(items) != expected_count:
        return False, f"Expected {expected_count} queries, got {len(items)}."
    for idx, query in enumerate(items, start=1):
        word_count = count_words(query)
        if word_count < 5 or word_count > 12:
            return False, f"Query {idx} has {word_count} words (expected 5-12)."
    return True, ""


def response_usage_tokens(response: object) -> Tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    in_tokens = 0
    out_tokens = 0

    # Responses API style
    in_tokens += int(getattr(usage, "input_tokens", 0) or 0)
    out_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    # Chat Completions style fallback
    in_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    if isinstance(usage, dict):
        in_tokens += int(usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0)
        out_tokens += int(usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0)
    return in_tokens, out_tokens


def build_prompt(template: str, passage: str) -> str:
    language_guard = (
        "IMPORTANT LANGUAGE RULE: Return all queries only in Azerbaijani language "
        "(Azeri, Latin script). Do not use English."
    )
    if "{news_passage}" in template:
        body = template.replace("{news_passage}", passage)
    else:
        body = f"{template}\n\nPASSAGE:\n{passage}"
    return f"{language_guard}\n\n{body}"


def iter_generate_queries(
    docs: Sequence[Dict[str, str]],
    template: str,
    model: str,
    queries_per_doc: int,
    max_passage_chars: int,
    max_output_tokens: int,
    max_retries: int,
    sleep_between_calls: float,
    checkpoint_path: Path | None = None,
    save_every_docs: int = 25,
) -> Tuple[List[Dict[str, str]], int, int]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: openai. Install with `pip install openai` "
            "or add it to requirements and reinstall."
        ) from exc

    client = OpenAI()
    all_queries: List[Dict[str, str]] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, article in enumerate(docs, start=1):
        doc_id = normalize_whitespace(str(article.get("id", "")))
        title = normalize_whitespace(str(article.get("title", "")))
        text = normalize_whitespace(str(article.get("text", "")))
        passage = sanitize_passage(title=title, text=text, max_chars=max_passage_chars)
        prompt = build_prompt(template=template, passage=passage)

        generated_for_doc: List[str] | None = None
        errors: List[str] = []
        for _attempt in range(1, max_retries + 1):
            try:
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=max_output_tokens,
                    reasoning={"effort": "minimal"},
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "query_list",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "queries": {
                                        "type": "array",
                                        "minItems": queries_per_doc,
                                        "maxItems": queries_per_doc,
                                        "items": {"type": "string"},
                                    }
                                },
                                "required": ["queries"],
                                "additionalProperties": False,
                            },
                        }
                    },
                )
                input_tokens, output_tokens = response_usage_tokens(response)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                status = str(getattr(response, "status", "") or "")
                incomplete = getattr(response, "incomplete_details", None)
                if status == "incomplete" and incomplete:
                    reason = getattr(incomplete, "reason", None)
                    if isinstance(incomplete, dict):
                        reason = incomplete.get("reason", reason)
                    errors.append(f"Incomplete response: {reason}")
                    continue

                items = extract_queries_from_response(response)
                valid, reason = validate_queries(items, expected_count=queries_per_doc)
                if not valid:
                    errors.append(reason)
                    continue

                generated_for_doc = items
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                if sleep_between_calls > 0:
                    time.sleep(sleep_between_calls)

        if generated_for_doc is None:
            print(f"[WARN] Skipped doc {doc_id} after retries. Last errors: {errors[-2:]}")
            continue

        for query in generated_for_doc:
            all_queries.append(
                {
                    "query_id": f"q{len(all_queries) + 1}",
                    "question": query,
                    "relevant_doc_id": doc_id,
                }
            )

        if i % 25 == 0:
            print(f"Processed docs: {i}/{len(docs)} | Generated queries: {len(all_queries)}")
        if checkpoint_path is not None and save_every_docs > 0 and i % save_every_docs == 0:
            save_queries(all_queries, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path} (queries={len(all_queries)})")
        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return all_queries, total_input_tokens, total_output_tokens


def save_queries(items: Iterable[Dict[str, str]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(items), f, ensure_ascii=False, indent=2)


def save_json(items: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def usd_cost(input_tokens: int, output_tokens: int, input_rate_per_1m: float, output_rate_per_1m: float) -> float:
    return (input_tokens / 1_000_000.0) * input_rate_per_1m + (output_tokens / 1_000_000.0) * output_rate_per_1m


def main() -> None:
    args = parse_args()
    load_dotenv(args.dotenv_path)
    if args.use_corpora_apa:
        args.news_path = DEFAULT_CORPORA_APA_PATH
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is not set. Put it in .env (OPENAI_API_KEY=...) "
            "or export it in your shell."
        )

    if args.reuse_selected_docs:
        docs = load_json_list(args.selected_docs_path)
    else:
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
            raise ValueError(f"No news loaded from: {args.news_path}")
        docs = select_docs(
            news=news,
            target_queries=args.target_queries,
            queries_per_doc=args.queries_per_doc,
            min_text_words=args.min_text_words,
            seed=args.seed,
        )

    save_json(docs, args.selected_docs_path)
    selected_doc_ids = [normalize_whitespace(str(item.get("id", ""))) for item in docs]
    save_json(selected_doc_ids, args.selected_doc_ids_path)
    print(f"Selected docs saved: {args.selected_docs_path}")
    print(f"Selected doc ids saved: {args.selected_doc_ids_path}")
    print(f"Selected doc slots: {len(docs)}")

    if args.selection_only:
        return

    prompt_template = load_prompt(args.prompt_path)
    generated, input_tokens, output_tokens = iter_generate_queries(
        docs=docs,
        template=prompt_template,
        model=args.model,
        queries_per_doc=args.queries_per_doc,
        max_passage_chars=args.max_passage_chars,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        sleep_between_calls=args.sleep_between_calls,
        checkpoint_path=args.output_path,
        save_every_docs=args.save_every_docs,
    )
    if not generated:
        raise RuntimeError("No queries generated.")

    generated = generated[: args.target_queries]
    save_queries(generated, args.output_path)

    cost = usd_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_rate_per_1m=args.input_rate_per_1m,
        output_rate_per_1m=args.output_rate_per_1m,
    )
    print(f"News source: {args.news_path}")
    print(f"Doc slots used: {len(docs)}")
    print(f"Generated queries: {len(generated)}")
    print(f"Saved to: {args.output_path}")
    print(f"Usage tokens (input/output): {input_tokens}/{output_tokens}")
    print(
        "Estimated generation cost (USD): "
        f"${cost:.4f} at rates input=${args.input_rate_per_1m}/1M, output=${args.output_rate_per_1m}/1M"
    )


if __name__ == "__main__":
    main()
