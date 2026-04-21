from __future__ import annotations

import math
from typing import Dict, List, Sequence


# Type aliases for readability
QueryId = str
ChunkId = str
Relevance = int

# qrels format:
# {
#   "q1": {"chunk_001": 2, "chunk_019": 1},
#   "q2": {"chunk_104": 2}
# }
Qrels = Dict[QueryId, Dict[ChunkId, Relevance]]

# results format:
# {
#   "q1": ["chunk_001", "chunk_050", "chunk_019", ...],
#   "q2": ["chunk_104", "chunk_111", ...]
# }
RunResults = Dict[QueryId, List[ChunkId]]


def precision_at_k(
    retrieved: Sequence[ChunkId],
    relevant: Dict[ChunkId, Relevance],
    k: int,
) -> float:
    """
    Precision@k = (# relevant items in top-k) / k

    We treat any qrel score > 0 as relevant for precision/recall.
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    top_k = list(retrieved[:k])
    if not top_k:
        return 0.0

    hits = sum(1 for chunk_id in top_k if relevant.get(chunk_id, 0) > 0)
    return hits / k


def recall_at_k(
    retrieved: Sequence[ChunkId],
    relevant: Dict[ChunkId, Relevance],
    k: int,
) -> float:
    """
    Recall@k = (# relevant items in top-k) / (total # relevant items)

    We treat any qrel score > 0 as relevant.
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    total_relevant = sum(1 for score in relevant.values() if score > 0)
    if total_relevant == 0:
        return 0.0

    top_k = list(retrieved[:k])
    hits = sum(1 for chunk_id in top_k if relevant.get(chunk_id, 0) > 0)
    return hits / total_relevant


def dcg_at_k(
    retrieved: Sequence[ChunkId],
    relevant: Dict[ChunkId, Relevance],
    k: int,
) -> float:
    """
    DCG@k using graded relevance:
    DCG = sum((2^rel - 1) / log2(rank + 1))

    rank is 1-based.
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        rel = relevant.get(chunk_id, 0)
        gain = (2**rel - 1) / math.log2(rank + 1)
        dcg += gain
    return dcg


def idcg_at_k(relevant: Dict[ChunkId, Relevance], k: int) -> float:
    """
    Ideal DCG@k computed by sorting true relevance labels descending.
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        gain = (2**rel - 1) / math.log2(rank + 1)
        idcg += gain
    return idcg


def ndcg_at_k(
    retrieved: Sequence[ChunkId],
    relevant: Dict[ChunkId, Relevance],
    k: int,
) -> float:
    """
    NDCG@k = DCG@k / IDCG@k
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    actual_dcg = dcg_at_k(retrieved, relevant, k)
    ideal_dcg = idcg_at_k(relevant, k)

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_query(
    retrieved: Sequence[ChunkId],
    relevant: Dict[ChunkId, Relevance],
    k: int = 5,
) -> Dict[str, float]:
    """
    Evaluate a single query.
    """
    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }


def evaluate_run(
    run_results: RunResults,
    qrels: Qrels,
    k: int = 5,
) -> Dict[str, object]:
    """
    Evaluate an entire run across all queries present in qrels.

    Returns:
    {
        "per_query": {
            "q1": {"precision@5": ..., "recall@5": ..., "ndcg@5": ...},
            ...
        },
        "aggregate": {
            "mean_precision@5": ...,
            "mean_recall@5": ...,
            "mean_ndcg@5": ...,
            "num_queries": ...
        }
    }
    """
    per_query: Dict[str, Dict[str, float]] = {}

    for query_id, relevant in qrels.items():
        retrieved = run_results.get(query_id, [])
        per_query[query_id] = evaluate_query(retrieved, relevant, k=k)

    num_queries = len(per_query)
    if num_queries == 0:
        return {
            "per_query": {},
            "aggregate": {
                f"mean_precision@{k}": 0.0,
                f"mean_recall@{k}": 0.0,
                f"mean_ndcg@{k}": 0.0,
                "num_queries": 0,
            },
        }

    mean_precision = sum(m[f"precision@{k}"] for m in per_query.values()) / num_queries
    mean_recall = sum(m[f"recall@{k}"] for m in per_query.values()) / num_queries
    mean_ndcg = sum(m[f"ndcg@{k}"] for m in per_query.values()) / num_queries

    return {
        "per_query": per_query,
        "aggregate": {
            f"mean_precision@{k}": mean_precision,
            f"mean_recall@{k}": mean_recall,
            f"mean_ndcg@{k}": mean_ndcg,
            "num_queries": num_queries,
        },
    }