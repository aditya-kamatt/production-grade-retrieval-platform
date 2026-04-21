from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from rank_bm25 import BM25Okapi


_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def simple_tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(slots=True)
class LexicalSearchResult:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict
    score: float
    rank: int
    source: str = "bm25"


class BM25Retriever:
    """
    Simple BM25 retriever over chunk texts.

    Why this design:
    - Keeps lexical search independent from vector search.
    - Easy to unit test because it is pure in-memory logic.
    - Lets you swap rank_bm25 with Elasticsearch/OpenSearch later
      without changing the search orchestration layer.
    """

    def __init__(self, chunks: Iterable[dict]) -> None:
        
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("BM25Retriever requires at least one chunk.")
        for chunk in self.chunks:
            if "text" not in chunk:
                raise ValueError("Chunk missing required field: text")

        self.tokenized_corpus = [simple_tokenize(chunk["text"]) for chunk in self.chunks]
        self.model = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> list[LexicalSearchResult]:
        if not query or not query.strip():
            return []

        tokenized_query = simple_tokenize(query)
        if not tokenized_query:
            return []

        scores = self.model.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        results: list[LexicalSearchResult] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self.chunks[idx]
            results.append(
                LexicalSearchResult(
                    chunk_id=str(chunk["chunk_id"]),
                    document_id=str(chunk["document_id"]),
                    text=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                    score=float(scores[idx]),
                    rank=rank,
                )
            )

        return results