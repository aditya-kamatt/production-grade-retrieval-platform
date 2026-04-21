from __future__ import annotations

from fastapi.testclient import TestClient
import pytest


class FakeSearchService:
    def __init__(self, *args, **kwargs) -> None:
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
        return {
            "query": query,
            "latency_ms": 12.34,
            "results": [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "text": "Result one",
                    "metadata": {"source": "doc1.txt"},
                    "rank": 1,
                    "scores": {
                        "bm25": 4.5,
                        "vector": 0.87,
                        "hybrid": 0.031,
                        "reranker": 0.95,
                    },
                },
                {
                    "chunk_id": "chunk-2",
                    "document_id": "doc-2",
                    "text": "Result two",
                    "metadata": {"source": "doc2.txt"},
                    "rank": 2,
                    "scores": {
                        "bm25": 3.8,
                        "vector": 0.82,
                        "hybrid": 0.029,
                        "reranker": 0.90,
                    },
                },
            ],
        }


@pytest.fixture
def client(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr("app.main.SearchService", FakeSearchService)

    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_and_root_endpoints(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    root = client.get("/")
    assert root.status_code == 200
    assert root.json() == {
        "message": "LEC Search API is running",
        "docs_url": "/docs",
        "search_endpoint": "/search",
    }


def test_search_endpoint_returns_expected_payload(client):
    response = client.post(
        "/search",
        json={
            "query": "machine learning",
            "candidate_k": 10,
            "final_k": 2,
            "use_reranker": True,
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "machine learning"
    assert len(body["results"]) == 2

    first = body["results"][0]
    assert first["chunk_id"] == "chunk-1"
    assert first["document_id"] == "doc-1"
    assert first["text"] == "Result one"
    assert first["metadata"] == {"source": "doc1.txt"}
    assert first["fused_score"] == 0.031
    assert first["rerank_score"] == 0.95
    assert first["component_scores"] == {
        "bm25": 4.5,
        "vector": 0.87,
        "hybrid": 0.031,
        "reranker": 0.95,
    }
    assert first["component_ranks"] == {}
    assert first["rank"] == 1
    assert first["latency_ms"] == 12.34


def test_search_endpoint_validation_error_when_candidate_k_less_than_final_k(client):
    response = client.post(
        "/search",
        json={
            "query": "machine learning",
            "candidate_k": 2,
            "final_k": 5,
            "use_reranker": True,
        },
    )

    assert response.status_code == 422