from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class RerankedSearchResult:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict
    rerank_score: float
    rank: int
    fused_score: float
    component_scores: dict[str, float]
    component_ranks: dict[str, int]


class CrossEncoderReranker:
    """
    Cross-encoder reranker over already-retrieved candidates.

    Why this design:
    - Expensive stage is isolated to small candidate sets.
    - Easy to disable for ablation studies in evaluation.
    - Lets you expose both fused_score and rerank_score in the API.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: Iterable, top_k: int = 5) -> list[RerankedSearchResult]:
        candidates = list(candidates)
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)

        reranked_rows = []
        for candidate, score in zip(candidates, scores):
            reranked_rows.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "document_id": candidate.document_id,
                    "text": candidate.text,
                    "metadata": candidate.metadata,
                    "rerank_score": float(score),
                    "fused_score": float(candidate.fused_score),
                    "component_scores": candidate.component_scores,
                    "component_ranks": candidate.component_ranks,
                }
            )

        reranked_rows.sort(key=lambda row: row["rerank_score"], reverse=True)
        reranked_rows = reranked_rows[:top_k]

        return [
            RerankedSearchResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                text=row["text"],
                metadata=row["metadata"],
                rerank_score=row["rerank_score"],
                rank=rank,
                fused_score=row["fused_score"],
                component_scores=row["component_scores"],
                component_ranks=row["component_ranks"],
            )
            for rank, row in enumerate(reranked_rows, start=1)
        ]