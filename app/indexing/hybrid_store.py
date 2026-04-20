from __future__ import annotations

from typing import List

from app.indexing.interfaces import ChunkRepository, EmbeddingRepository


class HybridRetrievalStore:
    def __init__(
        self,
        chunk_repository: ChunkRepository,
        embedding_repository: EmbeddingRepository,
    ) -> None:
        self.chunk_repository = chunk_repository
        self.embedding_repository = embedding_repository

    def semantic_search(self, query_vector: List[float], top_k: int = 5) -> List[dict]:
        vector_hits = self.embedding_repository.search(query_vector=query_vector, top_k=top_k)

        results = []
        for hit in vector_hits:
            chunk = self.chunk_repository.get_chunk_by_id(hit["chunk_id"])
            if chunk is None:
                continue

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "semantic_score": hit["score"],
                }
            )

        return results