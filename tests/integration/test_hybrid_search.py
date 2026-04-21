from __future__ import annotations

import numpy as np

import app.retrieval.reranker as reranker_module
from app.retrieval.hybrid_search import HybridSearchService
from app.retrieval.lexical_retriever import BM25Retriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import FaissVectorRetriever


class FakeCrossEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            text_lower = text.lower()
            if "hybrid retrieval combines bm25 and dense retrieval" in text_lower:
                scores.append(0.99)
            elif "embeddings capture semantic meaning" in text_lower:
                scores.append(0.80)
            else:
                scores.append(0.20)
        return scores


def fake_embedding_function(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        hash_val = hash(text) % 1000 / 1000.0
        vectors.append([hash_val, 1.0 - hash_val, 0.5])
    return np.array(vectors, dtype=np.float32)


def test_hybrid_pipeline_returns_ranked_results(monkeypatch):
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)

    chunks = [
        {
            "chunk_id": "1",
            "document_id": "doc-1",
            "text": "Hybrid retrieval combines BM25 and dense retrieval",
            "metadata": {"source": "a.txt"},
        },
        {
            "chunk_id": "2",
            "document_id": "doc-2",
            "text": "Embeddings capture semantic meaning",
            "metadata": {"source": "b.txt"},
        },
        {
            "chunk_id": "3",
            "document_id": "doc-3",
            "text": "Shipping invoices are processed daily",
            "metadata": {"source": "c.txt"},
        },
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0], 
            [0.8, 0.0, 0.0],  
            [0.0, 1.0, 0.0],  
        ],
        dtype=np.float32,
    )

    lexical = BM25Retriever(chunks)
    vector = FaissVectorRetriever(
        chunks=chunks,
        embeddings=embeddings,
        embedding_function=fake_embedding_function,
    )
    reranker = CrossEncoderReranker(model_name="fake-model")

    service = HybridSearchService(
        lexical_retriever=lexical,
        vector_retriever=vector,
        reranker=reranker,
    )

    results = service.search(
        query="how does hybrid retrieval work",
        candidate_k=3,
        final_k=2,
        use_reranker=True,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "1"
    assert "rerank_score" in results[0]
    assert "component_scores" in results[0]
    assert "component_ranks" in results[0]