from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import faiss
import numpy as np


@dataclass(slots=True)
class VectorSearchResult:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict
    score: float
    rank: int
    source: str = "dense"


class FaissVectorRetriever:
    """
    Exact dense retriever using FAISS IndexFlatIP.

    Assumptions:
    - You already have a sentence embedding model elsewhere.
    - Stored embeddings align 1:1 with `chunks`.
    - We normalize vectors so inner product behaves like cosine similarity.

    Why this design:
    - Exact baseline first; no premature IVF/HNSW complexity.
    - Clear separation between embedding model and ANN index.
    - Easy to replace with Qdrant/Weaviate later.
    """

    def __init__(
        self,
        chunks: Iterable[dict],
        embeddings: np.ndarray,
        embedding_function,
    ) -> None:
        self.chunks = list(chunks)
        self.embedding_function = embedding_function

        if len(self.chunks) == 0:
            raise ValueError("FaissVectorRetriever requires at least one chunk.")

        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings must be a numpy.ndarray")

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D: [num_chunks, embedding_dim]")

        if len(self.chunks) != embeddings.shape[0]:
            raise ValueError(
                "Mismatch between number of chunks and number of embeddings."
            )

        self.embeddings = embeddings.astype("float32", copy=True)
        faiss.normalize_L2(self.embeddings)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings) # type: ignore

    def _embed_query(self, query: str) -> np.ndarray:
        vector = self.embedding_function([query])

        if isinstance(vector, list):
            vector = np.asarray(vector)

        if not isinstance(vector, np.ndarray):
            raise TypeError("embedding_function must return a numpy array or list")

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        vector = vector.astype("float32", copy=False)
        faiss.normalize_L2(vector)
        return vector

    def search(self, query: str, top_k: int = 20) -> list[VectorSearchResult]:
        if not query or not query.strip():
            return []

        query_vector = self._embed_query(query)
        scores, indices = self.index.search(query_vector, top_k) # type: ignore

        results: list[VectorSearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue

            chunk = self.chunks[idx]
            results.append(
                VectorSearchResult(
                    chunk_id=str(chunk["chunk_id"]),
                    document_id=str(chunk["document_id"]),
                    text=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                    score=float(score),
                    rank=rank,
                )
            )

        return results