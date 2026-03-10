# RAG Retrieval Evaluation for Azerbaijani News

## Project Goal

This project evaluates the **retrieval** component of a Retrieval-Augmented Generation (RAG) pipeline for Azerbaijani news data.

It compares three retrieval approaches:

- BM25 lexical retrieval
- Embedding-based retrieval (Sentence Transformers + FAISS or Qdrant)
- Hybrid retrieval (weighted BM25 + embedding scores)

The project reports standard IR metrics:

- `Hit@K`
- `Recall@K`
- `Precision@K`
- `MRR` (Mean Reciprocal Rank)

## Project Structure

```text
rag_retrieval_eval/
    data/
        news.json
        queries.json

    indexing/
        chunk_news.py
        build_bm25_index.py
        build_embedding_index.py
        build_qdrant_index.py

    retrieval/
        bm25_retriever.py
        embedding_retriever.py
        qdrant_retriever.py
        hybrid_retriever.py

    evaluation/
        metrics.py
        evaluate.py

    experiments/
        run_experiments.py

    utils/
        text_utils.py

    README.md
    requirements.txt
```

## Dataset Format

`data/news.json`

```json
[
  {
    "id": "doc_1",
    "title": "....",
    "text": "...."
  }
]
```

Alternative source: `../Corpora/apa.az.csv` (from your parent `Corpora` folder).
Default CSV mapping used by the runner:

- `id` -> document id
- `title` -> title
- `content` -> body text
- optional metadata to payload/chunks: `link`, `date_time`

`data/queries.json`

```json
[
  {
    "query_id": "q1",
    "question": "....",
    "relevant_doc_id": "doc_123"
  }
]
```

## Retrieval Pipeline

1. News articles are chunked into about 300 tokens (`indexing/chunk_news.py`).
2. BM25 index is built on chunk text (`retrieval/bm25_retriever.py`).
3. Embedding index is built using:
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   - either:
     - FAISS `IndexFlatIP` with L2-normalized vectors (cosine similarity), or
     - Qdrant collection with cosine distance.
4. Hybrid retrieval combines normalized scores:

```text
hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_embedding
```

## How to Run

From the `rag_retrieval_eval` directory:

```bash
pip install -r requirements.txt
python experiments/run_experiments.py
```

Use the large APA corpus CSV as source:

```bash
python experiments/run_experiments.py \
  --use-corpora-apa \
  --auto-generate-queries \
  --news-limit 20000
```

Run with Qdrant backend for embeddings:

```bash
python experiments/run_experiments.py \
  --use-corpora-apa \
  --auto-generate-queries \
  --embedding-backend qdrant \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection rag_retrieval_eval_chunks \
  --qdrant-recreate-collection
```

If your environment has an older `protobuf` package and Qdrant import fails,
run:

```bash
pip install -U qdrant-client protobuf
```

Optional arguments:

```bash
python experiments/run_experiments.py \
  --k-values 5,10 \
  --hybrid-alpha 0.5 \
  --chunk-size 300 \
  --chunk-overlap 50 \
  --embedding-backend faiss \
  --visualize
```

## Output

Running experiments produces:

- printed result table in terminal
- `results/retrieval_results.csv`
- `results/query_level_results.csv`
- optional ASCII metric visualization (`--visualize`)

When `--embedding-backend qdrant` is used, chunk vectors are also persisted in Qdrant.
Payload fields ("nodes") per point:

- `doc_id`
- `chunk_id`
- `text`
- optional: `source_link`, `source_date`

This mapping allows retrieval hits to map back to the exact source chunk/document.

## Do You Need Qdrant For This Task?

Short answer: **not required** for offline retrieval evaluation.

- Use `faiss` (default) for local experiments, faster setup, no external service.
- Use `qdrant` when you need persistent/shared vector storage, multi-process access, or production-style retrieval infrastructure.

## Metrics Explanation

- `Hit@K`: 1 if the relevant document appears in top `K`, otherwise 0.
- `Precision@K`: number of relevant docs in top `K` divided by `K`.
- `Recall@K`: number of relevant docs in top `K` divided by total relevant docs.
- `MRR`: average reciprocal rank of the first relevant result.
  - rank = 1 -> score = 1.0
  - rank = 2 -> score = 0.5
  - rank = 5 -> score = 0.2

## Example Results Table

```text
Method      Hit@5   Hit@10   MRR
BM25        0.63    0.74     0.51
Embedding   0.71    0.83     0.62
Hybrid      0.79    0.89     0.68
```

These values are example numbers; your final results depend on your dataset and query set.


python experiments/run_experiments.py --use-corpora-apa --queries-path data/queries_merged.json --embedding-backend faiss --k-values 5,10 --hybrid-alpha 0.5 --model-name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2