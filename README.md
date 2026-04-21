# Production-Grade Retrieval Platform

A production-grade retrieval system designed with modular architecture,
hybrid search, and measurable evaluation.

This system ingests heterogeneous documents, processes them into
structured chunks, builds lexical and vector indexes, and serves ranked
search results via API and CLI.



## Features

-   Hybrid retrieval (BM25 + vector search)
-   Optional reranking layer
-   Multi-format ingestion (PDF, CSV, TXT, Markdown)
-   Modular pipeline (ingestion → processing → indexing → retrieval)
-   FastAPI-based search API
-   CLI interface for quick testing
-   Evaluation framework:
    -   Precision@5
    -   Recall@5
    -   NDCG@5
-   Docker support
-   Full unit + integration tests



## Architecture Overview

Pipeline:

1.  **Discovery** → finds raw documents\
2.  **Extraction** → parses documents into text\
3.  **Normalisation** → cleans text\
4.  **Chunking** → splits into retrieval units\
5.  **Indexing** →
    -   FAISS (vector)\
    -   SQLite / BM25 (lexical)\
6.  **Retrieval** →
    -   lexical\
    -   semantic\
    -   hybrid\
    -   hybrid + rerank\
7.  **Serving** → FastAPI + CLI



## Project Structure

    app/
      api/            # API routes
      core/           # search service
      ingestion/      # ingestion pipeline
      processing/     # chunking + normalisation
      indexing/       # FAISS + SQLite
      retrieval/      # retrievers + reranker
      evaluation/     # metrics + runner
      scripts/        # CLI tools

    data/
      raw/            # input documents
      processed/      # generated indexes
      evaluation/     # queries + qrels + runs

    tests/
      unit/
      integration/



## Setup

``` bash
git clone https://github.com/aditya-kamatt/production-grade-retrieval-platform.git
cd production-grade-retrieval-platform
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



## Run Ingestion

``` bash
python app/scripts/run_ingestion.py
```

This: - extracts documents - chunks text - builds embeddings - writes
indexes


## Run API

``` bash
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000/docs


## Example Request

    POST /search

``` json
{
  "query": "What is hybrid search?",
  "top_k": 5
}
```


## CLI Search

``` bash
python app/scripts/search_cli.py
```


## Run Tests

``` bash
pytest
```

## Evaluation

Located in:

    app/evaluation/
    data/evaluation/

Run:

``` bash
python app/evaluation/runner.py
```

Metrics: - Precision@5 - Recall@5 - NDCG@5

Compare: - semantic-only - hybrid - hybrid + rerank



## Docker

``` bash
docker build -t retrieval-platform .
docker run -p 8000:8000 retrieval-platform
```


## Design Decisions

-   Hybrid search improves robustness over pure lexical/semantic
-   Modular design allows independent testing and tuning
-   Evaluation-first mindset ensures measurable correctness



## Known Limitations

-   ingestion required before search
-   retrieval quality depends on chunking + qrels
-   reranking increases latency



## Cleanup Before Submission

Ensure these are NOT committed:

    __pycache__/
    *.pyc
    data/processed/
    data/evaluation/runs/
    data/evaluation/reports/



## Submission Checklist

-   [ ] README complete
-   [ ] tests passing
-   [ ] ingestion works from scratch
-   [ ] API runs
-   [ ] evaluation reproducible
-   [ ] docs added (report, architecture, roadmap, AI usage)



## Future Improvements

-   chunk-size experimentation
-   caching embeddings
-   latency monitoring
-   better evaluation reporting
-   production observability



## License

See LICENSE file.
