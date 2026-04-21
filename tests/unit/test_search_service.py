from __future__ import annotations

from types import SimpleNamespace

import pytest
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
                "metadata": {"source": "doc1"},
            },
            {
                "chunk_id": "c2",
                "doc_id": "d2",
                "text": "bm25 retrieval methods",
                "metadata": {"source": "doc2"},
            },
        ]


class EmptyChunkRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def get_all_chunks(self):
        return []


class FakeEmbeddingRepository:
    def __init__(self, index_path: str, id_map_path: str, embedding_dimension: int) -> None:
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.embedding_dimension = embedding_dimension
        self.search_calls = []

    def search(self, query_vec, top_k: int):
        self.search_calls.append((query_vec, top_k))
        return [
            {"chunk_id": "c2", "score": 0.91},
            {"chunk_id": "c1", "score": 0.76},
        ]

class FakeBM25Retriever:
    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.search_calls = []

    def search(self, query: str, candidate_k: int):
        self.search_calls.append((query, candidate_k))
        return [
            SimpleNamespace(
                chunk_id="c1",
                document_id="d1",
                text="machine learning basics",
                metadata={"source": "doc1"},
                score=8.5,
                rank=1,
            ),
            SimpleNamespace(
                chunk_id="c2",
                document_id="d2",
                text="bm25 retrieval methods",
                metadata={"source": "doc2"},
                score=6.2,
                rank=2,
            ),
        ]


class FakeReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
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
                text="bm25 retrieval methods",
                metadata={"source": "doc2"},
                hybrid_score=0.0320,
                rerank_score=0.98,
                component_scores={"bm25": 6.2, "vector": 0.91},
            ),
            SimpleNamespace(
                chunk_id="c1",
                document_id="d1",
                text="machine learning basics",
                metadata={"source": "doc1"},
                hybrid_score=0.0318,
                rerank_score=0.87,
                component_scores={"bm25": 8.5, "vector": 0.76},
            ),
        ]


@pytest.fixture
def patched_search_service(monkeypatch):
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

    return SearchService


def test_init_loads_chunks_and_builds_chunk_lookup(patched_search_service):
    service = patched_search_service()

    assert len(service.chunks) == 2
    assert "c1" in service.chunk_by_id
    assert "c2" in service.chunk_by_id

    # SearchService mutates each chunk to include document_id = doc_id
    assert service.chunk_by_id["c1"]["document_id"] == "d1"
    assert service.chunk_by_id["c2"]["document_id"] == "d2"


def test_init_raises_when_no_chunks(monkeypatch):
    from app.core.search_service import SearchService

    monkeypatch.setattr(
        "app.core.search_service.SQLiteChunkRepository",
        EmptyChunkRepository,
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

    with pytest.raises(ValueError, match="No chunks found. Run ingestion first."):
        SearchService()


def test_embed_returns_float_list(patched_search_service):
    service = patched_search_service()

    embedding = service._embed("what is bm25")

    assert embedding == [1.0, 2.0, 3.0]
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)


def test_vector_search_returns_ranked_chunk_results(patched_search_service):
    service = patched_search_service()

    results = service._vector_search("retrieval", top_k=2)

    assert len(results) == 2

    assert results[0]["chunk_id"] == "c2"
    assert results[0]["document_id"] == "d2"
    assert results[0]["score"] == 0.91
    assert results[0]["rank"] == 1

    assert results[1]["chunk_id"] == "c1"
    assert results[1]["document_id"] == "d1"
    assert results[1]["score"] == 0.76
    assert results[1]["rank"] == 2


def test_vector_search_skips_missing_chunk_ids(patched_search_service):
    service = patched_search_service()

    def fake_search(query_vec, top_k):
        return [
            {"chunk_id": "missing", "score": 0.99},
            {"chunk_id": "c1", "score": 0.76},
        ]

    service.embedding_repository.search = fake_search

    results = service._vector_search("retrieval", top_k=2)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    # rank comes from enumerate over hits, so it keeps original ranking position
    assert results[0]["rank"] == 2


def test_fuse_combines_lexical_and_vector_scores_and_sorts_descending(patched_search_service):
    from app.core.search_service import HybridCandidate
    service = patched_search_service()

    lexical = [
        SimpleNamespace(
            chunk_id="c1",
            document_id="d1",
            text="machine learning basics",
            metadata={"source": "doc1"},
            score=8.5,
            rank=1,
        )
    ]
    vector = [
        {
            "chunk_id": "c2",
            "document_id": "d2",
            "text": "bm25 retrieval methods",
            "metadata": {"source": "doc2"},
            "score": 0.91,
            "rank": 1,
        },
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "text": "machine learning basics",
            "metadata": {"source": "doc1"},
            "score": 0.76,
            "rank": 2,
        },
    ]

    fused = service._fuse(lexical, vector, k=5)

    assert len(fused) == 2
    assert all(isinstance(item, HybridCandidate) for item in fused)

    # c1 appears in both lexical and vector, so it should win
    assert fused[0].chunk_id == "c1"
    assert fused[0].component_scores["bm25"] == 8.5
    assert fused[0].component_scores["vector"] == 0.76
    assert fused[0].component_ranks["bm25"] == 1
    assert fused[0].component_ranks["vector"] == 2

    # c2 appears only in vector
    assert fused[1].chunk_id == "c2"
    assert fused[1].component_scores["bm25"] == 0.0
    assert fused[1].component_scores["vector"] == 0.91
    assert fused[1].component_ranks["vector"] == 1


def test_search_with_reranker_returns_expected_shape_and_scores(patched_search_service):
    service = patched_search_service()

    result = service.search(
        query="bm25",
        candidate_k=10,
        final_k=2,
        use_reranker=True,
    )

    assert result["query"] == "bm25"
    assert isinstance(result["latency_ms"], float)
    assert len(result["results"]) == 2

    first = result["results"][0]
    assert first["chunk_id"] == "c2"
    assert first["document_id"] == "d2"
    assert first["rank"] == 1
    assert first["scores"]["bm25"] == 6.2
    assert first["scores"]["vector"] == 0.91
    assert first["scores"]["hybrid"] == 0.0320
    assert first["scores"]["reranker"] == 0.98


def test_search_without_reranker_returns_fused_results_and_none_reranker_score(patched_search_service):
    service = patched_search_service()

    result = service.search(
        query="bm25",
        candidate_k=10,
        final_k=2,
        use_reranker=False,
    )

    assert result["query"] == "bm25"
    assert isinstance(result["latency_ms"], float)
    assert len(result["results"]) == 2

    first = result["results"][0]
    second = result["results"][1]

    assert first["rank"] == 1
    assert second["rank"] == 2

    assert first["scores"]["reranker"] is None
    assert second["scores"]["reranker"] is None

    assert "bm25" in first["scores"]
    assert "vector" in first["scores"]
    assert "hybrid" in first["scores"]


def test_search_passes_expected_arguments_to_retriever_and_reranker(patched_search_service):
    service = patched_search_service()

    result = service.search(
        query="vector databases",
        candidate_k=7,
        final_k=3,
        use_reranker=True,
    )

    assert result["query"] == "vector databases"

    assert service.lexical_retriever.search_calls == [("vector databases", 7)]
    assert len(service.reranker.calls) == 1
    assert service.reranker.calls[0]["query"] == "vector databases"
    assert service.reranker.calls[0]["top_k"] == 3
    assert len(service.reranker.calls[0]["candidates"]) >= 1