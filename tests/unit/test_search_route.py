from __future__ import annotations

from types import SimpleNamespace

from app.api.routes.search import search_endpoint


class FakeSearchService:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = []

    def search(self, query: str, candidate_k: int, final_k: int, use_reranker: bool):
        self.calls.append(
            {
                "query": query,
                "candidate_k": candidate_k,
                "final_k": final_k,
                "use_reranker": use_reranker,
            }
        )
        return self.response


def test_search_endpoint_maps_service_response_into_schema():
    service_response = {
        "query": "machine learning",
        "latency_ms": 12.345,
        "results": [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "text": "Machine learning is a field of AI.",
                "metadata": {"source": "txt_ai_intro_01.txt"},
                "rank": 1,
                "scores": {
                    "bm25": 4.2,
                    "vector": 0.88,
                    "hybrid": 0.0321,
                    "reranker": 0.97,
                },
            },
            {
                "chunk_id": "chunk-2",
                "document_id": "doc-2",
                "text": "Supervised learning uses labelled data.",
                "metadata": {"source": "txt_machine_learning_04.txt"},
                "rank": 2,
                "scores": {
                    "bm25": 3.9,
                    "vector": 0.81,
                    "hybrid": 0.0315,
                    "reranker": 0.91,
                },
            },
        ],
    }

    fake_service = FakeSearchService(service_response)
    request = SimpleNamespace(
        query="machine learning",
        candidate_k=20,
        final_k=5,
        use_reranker=True,
    )

    response = search_endpoint(request=request, search_service=fake_service)

    assert fake_service.calls == [
        {
            "query": "machine learning",
            "candidate_k": 20,
            "final_k": 5,
            "use_reranker": True,
        }
    ]

    assert response.query == "machine learning"
    assert len(response.results) == 2

    first = response.results[0]
    assert first.chunk_id == "chunk-1"
    assert first.document_id == "doc-1"
    assert first.text == "Machine learning is a field of AI."
    assert first.metadata == {"source": "txt_ai_intro_01.txt"}
    assert first.fused_score == 0.0321
    assert first.rerank_score == 0.97
    assert first.component_scores == {
        "bm25": 4.2,
        "vector": 0.88,
        "hybrid": 0.0321,
        "reranker": 0.97,
    }
    assert first.component_ranks == {}
    assert first.rank == 1
    assert first.latency_ms == 12.345


def test_search_endpoint_supports_missing_metadata_with_default_empty_dict():
    service_response = {
        "query": "bm25",
        "latency_ms": 9.5,
        "results": [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "text": "BM25 is a lexical ranking function.",
                "rank": 1,
                "scores": {
                    "bm25": 7.1,
                    "vector": 0.0,
                    "hybrid": 0.0164,
                    "reranker": None,
                },
            }
        ],
    }

    fake_service = FakeSearchService(service_response)
    request = SimpleNamespace(
        query="bm25",
        candidate_k=10,
        final_k=3,
        use_reranker=False,
    )

    response = search_endpoint(request=request, search_service=fake_service)

    assert response.query == "bm25"
    assert len(response.results) == 1
    assert response.results[0].metadata == {}
    assert response.results[0].rerank_score is None
    assert response.results[0].component_scores["reranker"] == 0.0
    assert response.results[0].latency_ms == 9.5


def test_search_endpoint_applies_same_latency_to_each_result():
    service_response = {
        "query": "vector search",
        "latency_ms": 21.123,
        "results": [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "text": "Result one",
                "metadata": {},
                "rank": 1,
                "scores": {
                    "bm25": 1.0,
                    "vector": 0.9,
                    "hybrid": 0.02,
                    "reranker": 0.8,
                },
            },
            {
                "chunk_id": "c2",
                "document_id": "d2",
                "text": "Result two",
                "metadata": {},
                "rank": 2,
                "scores": {
                    "bm25": 0.8,
                    "vector": 0.7,
                    "hybrid": 0.019,
                    "reranker": 0.75,
                },
            },
        ],
    }

    fake_service = FakeSearchService(service_response)
    request = SimpleNamespace(
        query="vector search",
        candidate_k=15,
        final_k=2,
        use_reranker=True,
    )

    response = search_endpoint(request=request, search_service=fake_service)

    assert response.results[0].latency_ms == 21.123
    assert response.results[1].latency_ms == 21.123