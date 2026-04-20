from __future__ import annotations

import numpy as np

from app.retrieval.vector_retriever import FaissVectorRetriever


def fake_embedding_function(texts: list[str]) -> np.ndarray:
    mapping = {
        "hybrid retrieval": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "export logistics": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "reranking": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    return np.vstack([mapping[text] for text in texts]).astype(np.float32)


def test_vector_retriever_returns_expected_nearest_neighbor():
    chunks = [
        {
            "chunk_id": "1",
            "document_id": "doc-1",
            "text": "Hybrid chunk",
            "metadata": {},
        },
        {
            "chunk_id": "2",
            "document_id": "doc-2",
            "text": "Logistics chunk",
            "metadata": {},
        },
        {
            "chunk_id": "3",
            "document_id": "doc-3",
            "text": "Reranking chunk",
            "metadata": {},
        },
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # chunk 1
            [0.0, 1.0, 0.0],  # chunk 2
            [0.0, 0.0, 1.0],  # chunk 3
        ],
        dtype=np.float32,
    )

    retriever = FaissVectorRetriever(
        chunks=chunks,
        embeddings=embeddings,
        embedding_function=fake_embedding_function,
    )

    results = retriever.search("hybrid retrieval", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "1"
    assert results[0].rank == 1
    assert results[0].score >= results[1].score


def test_vector_retriever_empty_query_returns_empty_list():
    chunks = [
        {
            "chunk_id": "1",
            "document_id": "doc-1",
            "text": "Anything",
            "metadata": {},
        }
    ]
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    retriever = FaissVectorRetriever(
        chunks=chunks,
        embeddings=embeddings,
        embedding_function=fake_embedding_function,
    )

    assert retriever.search("", top_k=5) == []