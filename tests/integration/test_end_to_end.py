from __future__ import annotations

import pytest

from types import SimpleNamespace



class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, texts, convert_to_numpy: bool, normalize_embeddings: bool):
        import numpy as np

        self.calls.append(
            {
                "texts": texts,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

class FakeChunkRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def get_all_chunks(self):
        return [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "text": "machine learning basics",
                "metadata": {"source": "ml.txt"},
            },
            {
                "chunk_id": "c2",
                "doc_id": "d2",
                "text": "bm25 ranking function",
                "metadata": {"source": "bm25.txt"},
            },
            {
                "chunk_id": "c3",
                "doc_id": "d3",
                "text": "vector search embeddings",
                "metadata": {"source": "vector.txt"},
            },
        ]


class FakeEmbeddingRepository:
    def __init__(self, index_path: str, id_map_path: str, embedding_dimension: int) -> None:
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.embedding_dimension = embedding_dimension

    def search(self, query_vec, top_k: int):
        return [
            {"chunk_id": "c3", "score": 0.92},
            {"chunk_id": "c1", "score": 0.81},
            {"chunk_id": "c2", "score": 0.77},
        ][:top_k]

class FakeBM25Retriever:
    def __init__(self, chunks) -> None:
        self.chunks = chunks

    def search(self, query: str, candidate_k: int):
        return [
            SimpleNamespace(
                chunk_id="c2",
                document_id="d2",
                text="bm25 ranking function",
                metadata={"source": "bm25.txt"},
                score=8.9,
                rank=1,
            ),
            SimpleNamespace(
                chunk_id="c1",
                document_id="d1",
                text="machine learning basics",
                metadata={"source": "ml.txt"},
                score=6.3,
                rank=2,
            ),
        ][:candidate_k]


class FakeReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def rerank(self, query: str, candidates, top_k: int):
        ranked = [
            SimpleNamespace(
                chunk_id="c2",
                document_id="d2",
                text="bm25 ranking function",
                metadata={"source": "bm25.txt"},
                hybrid_score=0.0325,
                rerank_score=0.99,
                component_scores={"bm25": 8.9, "vector": 0.77},
            ),
            SimpleNamespace(
                chunk_id="c3",
                document_id="d3",
                text="vector search embeddings",
                metadata={"source": "vector.txt"},
                hybrid_score=0.0164,
                rerank_score=0.91,
                component_scores={"bm25": 0.0, "vector": 0.92},
            ),
            SimpleNamespace(
                chunk_id="c1",
                document_id="d1",
                text="machine learning basics",
                metadata={"source": "ml.txt"},
                hybrid_score=0.0320,
                rerank_score=0.85,
                component_scores={"bm25": 6.3, "vector": 0.81},
            ),
        ]
        return ranked[:top_k]


@pytest.fixture
def service(monkeypatch):
    from app.core.search_service import SearchService
    
    monkeypatch.setattr(
        "app.core.search_service.SQLiteChunkRepository",
        FakeChunkRepository,
    )
    monkeypatch.setattr(
        "app.core.search_service.FAISSEmbeddingRepository",
        FakeEmbeddingRepository,
    )
    monkeypatch.setattr(
    SearchService,
    "_load_embedding_model",
    lambda self, model_name: FakeEmbeddingModel(),
)
    monkeypatch.setattr(
        "app.core.search_service.BM25Retriever",
        FakeBM25Retriever,
    )
    monkeypatch.setattr(
        "app.core.search_service.CrossEncoderReranker",
        FakeReranker,
    )

    return SearchService()


def test_end_to_end_search_without_reranker(service):
    result = service.search(
        query="search methods",
        candidate_k=3,
        final_k=2,
        use_reranker=False,
    )

    assert result["query"] == "search methods"
    assert isinstance(result["latency_ms"], float)
    assert len(result["results"]) == 2

    for idx, item in enumerate(result["results"], start=1):
        assert item["rank"] == idx
        assert "chunk_id" in item
        assert "document_id" in item
        assert "text" in item
        assert "metadata" in item
        assert "scores" in item
        assert item["scores"]["reranker"] is None


def test_end_to_end_search_with_reranker(service):
    result = service.search(
        query="search methods",
        candidate_k=3,
        final_k=2,
        use_reranker=True,
    )

    assert result["query"] == "search methods"
    assert isinstance(result["latency_ms"], float)
    assert len(result["results"]) == 2

    first = result["results"][0]
    second = result["results"][1]

    assert first["chunk_id"] == "c2"
    assert first["rank"] == 1
    assert first["scores"]["reranker"] == 0.99

    assert second["chunk_id"] == "c3"
    assert second["rank"] == 2
    assert second["scores"]["reranker"] == 0.91