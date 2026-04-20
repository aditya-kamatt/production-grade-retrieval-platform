from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HybridSearchResult:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    hybrid_score: float
    rank: int
    component_scores: dict[str, float] = field(default_factory=dict)
    component_ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_hybrid(
    result_sets: dict[str, list],
    top_k: int = 20,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[HybridSearchResult]:
    """
    Fuse multiple ranked result lists with weighted Reciprocal Rank Fusion.

    Parameters
    ----------
    result_sets:
        Mapping like:
        {
            "bm25": [LexicalSearchResult, ...],
            "dense": [VectorSearchResult, ...]
        }

    top_k:
        Final number of results returned.

    k:
        RRF damping constant. 60 is a common default.

    weights:
        Optional per-source weights, e.g.
        {"bm25": 1.0, "dense": 1.0}
    """
    if weights is None:
        weights = {name: 1.0 for name in result_sets.keys()}

    aggregated: dict[str, dict] = {}

    for source_name, results in result_sets.items():
        source_weight = weights.get(source_name, 1.0)

        for item in results:
            chunk_id = item.chunk_id
            if chunk_id not in aggregated:
                aggregated[chunk_id] = {
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "text": item.text,
                    "metadata": item.metadata,
                    "hybrid_score": 0.0,
                    "component_scores": {},
                    "component_ranks": {},
                }

            aggregated[chunk_id]["hybrid_score"] += source_weight * (1.0 / (k + item.rank))
            aggregated[chunk_id]["component_scores"][source_name] = float(item.score)
            aggregated[chunk_id]["component_ranks"][source_name] = int(item.rank)

    ranked = sorted(
        aggregated.values(),
        key=lambda row: row["hybrid_score"],
        reverse=True,
    )[:top_k]

    return [
        HybridSearchResult(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            text=row["text"],
            metadata=row["metadata"],
            hybrid_score=float(row["hybrid_score"]),
            rank=rank,
            component_scores=row["component_scores"],
            component_ranks=row["component_ranks"],
        )
        for rank, row in enumerate(ranked, start=1)
    ]


def minmax_score_hybrid(
    result_sets: dict[str, list],
    top_k: int = 20,
    weights: dict[str, float] | None = None,
) -> list[HybridSearchResult]:
    """
    Optional alternative when you explicitly want score-based fusion.

    Use this only for experimentation in evaluation, because
    normalized-score fusion is usually less robust than RRF.
    """
    if weights is None:
        weights = {name: 1.0 for name in result_sets.keys()}

    aggregated: dict[str, dict] = {}

    for source_name, results in result_sets.items():
        if not results:
            continue

        raw_scores = [float(item.score) for item in results]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        denom = max(max_score - min_score, 1e-12)
        source_weight = weights.get(source_name, 1.0)

        for item in results:
            normalised = (float(item.score) - min_score) / denom
            chunk_id = item.chunk_id

            if chunk_id not in aggregated:
                aggregated[chunk_id] = {
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "text": item.text,
                    "metadata": item.metadata,
                    "hybrid_score": 0.0,
                    "component_scores": {},
                    "component_ranks": {},
                }

            aggregated[chunk_id]["hybrid_score"] += source_weight * normalised
            aggregated[chunk_id]["component_scores"][source_name] = float(item.score)
            aggregated[chunk_id]["component_ranks"][source_name] = int(item.rank)

    ranked = sorted(
        aggregated.values(),
        key=lambda row: row["hybrid_score"],
        reverse=True,
    )[:top_k]

    return [
        HybridSearchResult(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            text=row["text"],
            metadata=row["metadata"],
            hybrid_score=float(row["hybrid_score"]),
            rank=rank,
            component_scores=row["component_scores"],
            component_ranks=row["component_ranks"],
        )
        for rank, row in enumerate(ranked, start=1)
    ]