from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.retrieval.hybrid_search import HybridSearchService


# ---------- Fakes ----------

class FakeLexicalRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return ["lexical_results"]


class FakeVectorRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return ["vector_results"]


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query: str, candidates, top_k: int):
        self.calls.append(
            {
                "query": query,
                "candidates": candidates,
                "top_k": top_k,
            }
        )
        return [
            SimpleNamespace(
                chunk_id="c2",
                document_id="d2",
                text="result 2",
                metadata={},
                hybrid_score=0.03,
                rerank_score=0.95,
                component_scores={"bm25": 5.0, "dense": 0.9},
                component_ranks={"bm25": 1, "dense": 2},
                rank=1,
            ),
            SimpleNamespace(
                chunk_id="c1",
                document_id="d1",
                text="result 1",
                metadata={},
                hybrid_score=0.02,
                rerank_score=0.85,
                component_scores={"bm25": 4.0, "dense": 0.8},
                component_ranks={"bm25": 2, "dense": 1},
                rank=2,
            ),
        ]


# ---------- Tests ----------

def test_search_calls_retrievers_and_fusion(monkeypatch):
    fake_lexical = FakeLexicalRetriever()
    fake_vector = FakeVectorRetriever()

    # Patch fusion function
    def fake_rrf(result_sets, top_k, weights):
        assert "bm25" in result_sets
        assert "dense" in result_sets
        assert weights == {"bm25": 1.0, "dense": 1.0}

        return [
            SimpleNamespace(
                chunk_id="c1",
                document_id="d1",
                text="result 1",
                metadata={},
                hybrid_score=0.02,
                component_scores={"bm25": 4.0, "dense": 0.8},
                component_ranks={"bm25": 1, "dense": 1},
            )
        ]

    monkeypatch.setattr(
        "app.retrieval.hybrid_search.reciprocal_rank_hybrid",
        fake_rrf,
    )

    service = HybridSearchService(
        lexical_retriever=fake_lexical,
        vector_retriever=fake_vector,
        reranker=None,
    )

    results = service.search(query="test", candidate_k=5, final_k=1)

    assert fake_lexical.calls == [("test", 5)]
    assert fake_vector.calls == [("test", 5)]

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rerank_score"] is None
    assert results[0]["rank"] == 1


def test_search_with_reranker(monkeypatch):
    fake_lexical = FakeLexicalRetriever()
    fake_vector = FakeVectorRetriever()
    fake_reranker = FakeReranker()

    def fake_rrf(result_sets, top_k, weights):
        return ["fused_candidates"]

    monkeypatch.setattr(
        "app.retrieval.hybrid_search.reciprocal_rank_hybrid",
        fake_rrf,
    )

    service = HybridSearchService(
        lexical_retriever=fake_lexical,
        vector_retriever=fake_vector,
        reranker=fake_reranker,
    )

    results = service.search(
        query="test",
        candidate_k=5,
        final_k=2,
        use_reranker=True,
    )

    assert len(fake_reranker.calls) == 1
    assert fake_reranker.calls[0]["query"] == "test"
    assert fake_reranker.calls[0]["top_k"] == 2

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["rerank_score"] == 0.95
    assert results[0]["rank"] == 1


def test_search_without_reranker_even_if_present(monkeypatch):
    fake_lexical = FakeLexicalRetriever()
    fake_vector = FakeVectorRetriever()
    fake_reranker = FakeReranker()

    def fake_rrf(result_sets, top_k, weights):
        return [
            SimpleNamespace(
                chunk_id="c1",
                document_id="d1",
                text="result 1",
                metadata={},
                hybrid_score=0.02,
                component_scores={},
                component_ranks={},
            )
        ]

    monkeypatch.setattr(
        "app.retrieval.hybrid_search.reciprocal_rank_hybrid",
        fake_rrf,
    )

    service = HybridSearchService(
        lexical_retriever=fake_lexical,
        vector_retriever=fake_vector,
        reranker=fake_reranker,
    )

    results = service.search(
        query="test",
        candidate_k=5,
        final_k=1,
        use_reranker=False,
    )

    # Reranker should NOT be called
    assert fake_reranker.calls == []

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rerank_score"] is None
    assert results[0]["rank"] == 1


def test_search_uses_custom_weights(monkeypatch):
    fake_lexical = FakeLexicalRetriever()
    fake_vector = FakeVectorRetriever()

    captured_weights = {}

    def fake_rrf(result_sets, top_k, weights):
        captured_weights.update(weights)
        return []

    monkeypatch.setattr(
        "app.retrieval.hybrid_search.reciprocal_rank_hybrid",
        fake_rrf,
    )

    service = HybridSearchService(
        lexical_retriever=fake_lexical,
        vector_retriever=fake_vector,
        reranker=None,
    )

    service.search(
        query="test",
        candidate_k=5,
        final_k=1,
        hybrid_weights={"bm25": 2.0, "dense": 0.5},
    )

    assert captured_weights == {"bm25": 2.0, "dense": 0.5}