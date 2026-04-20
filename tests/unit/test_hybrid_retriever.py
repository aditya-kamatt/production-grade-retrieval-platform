from dataclasses import dataclass

from app.retrieval.hybrid_retriever import minmax_score_hybrid, reciprocal_rank_hybrid


@dataclass
class DummyResult:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict
    score: float
    rank: int


def test_rrf_hybrid_combines_two_result_sets():
    bm25_results = [
        DummyResult("1", "doc-1", "text-1", {}, 10.0, 1),
        DummyResult("2", "doc-2", "text-2", {}, 8.0, 2),
    ]
    dense_results = [
        DummyResult("2", "doc-2", "text-2", {}, 0.95, 1),
        DummyResult("3", "doc-3", "text-3", {}, 0.90, 2),
    ]

    fused = reciprocal_rank_hybrid(
        result_sets={"bm25": bm25_results, "dense": dense_results},
        top_k=3,
    )

    assert len(fused) == 3
    fused_ids = [row.chunk_id for row in fused]
    assert set(fused_ids) == {"1", "2", "3"}

    # chunk 2 appears in both lists, so it should rank strongly
    assert fused[0].chunk_id == "2"


def test_rrf_preserves_component_scores_and_ranks():
    bm25_results = [DummyResult("1", "doc-1", "text-1", {}, 5.0, 1)]
    dense_results = [DummyResult("1", "doc-1", "text-1", {}, 0.9, 2)]

    fused = reciprocal_rank_hybrid(
        result_sets={"bm25": bm25_results, "dense": dense_results},
        top_k=1,
    )

    assert fused[0].component_scores["bm25"] == 5.0
    assert fused[0].component_scores["dense"] == 0.9
    assert fused[0].component_ranks["bm25"] == 1
    assert fused[0].component_ranks["dense"] == 2


def test_minmax_score_hybrid_runs_and_returns_ranked_output():
    bm25_results = [
        DummyResult("1", "doc-1", "text-1", {}, 10.0, 1),
        DummyResult("2", "doc-2", "text-2", {}, 5.0, 2),
    ]
    dense_results = [
        DummyResult("2", "doc-2", "text-2", {}, 0.9, 1),
        DummyResult("3", "doc-3", "text-3", {}, 0.1, 2),
    ]

    fused = minmax_score_hybrid(
        result_sets={"bm25": bm25_results, "dense": dense_results},
        top_k=3,
    )

    assert len(fused) == 3
    assert all(row.rank >= 1 for row in fused)