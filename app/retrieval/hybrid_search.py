from __future__ import annotations

from app.retrieval.hybrid_retriever import reciprocal_rank_hybrid 
from app.retrieval.lexical_retriever import BM25Retriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import FaissVectorRetriever


class HybridSearchService:
    def __init__(
        self,
        lexical_retriever: BM25Retriever,
        vector_retriever: FaissVectorRetriever,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.lexical_retriever = lexical_retriever
        self.vector_retriever = vector_retriever
        self.reranker = reranker

    def search(
        self,
        query: str,
        candidate_k: int = 20,
        final_k: int = 5,
        hybrid_weights: dict[str, float] | None = None,
        use_reranker: bool = True,
    ) -> list[dict]:
        lexical_results = self.lexical_retriever.search(query=query, top_k=candidate_k)
        dense_results = self.vector_retriever.search(query=query, top_k=candidate_k)

        fused_results = reciprocal_rank_hybrid(
            result_sets={
                "bm25": lexical_results,
                "dense": dense_results,
            },
            top_k=candidate_k,
            weights=hybrid_weights or {"bm25": 1.0, "dense": 1.0},
        )

        if use_reranker and self.reranker is not None:
            reranked = self.reranker.rerank(
                query=query,
                candidates=fused_results,
                top_k=final_k,
            )
            return [
                {
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "text": item.text,
                    "metadata": item.metadata,
                    "hybrid_score": item.hybrid_score,
                    "rerank_score": item.rerank_score,
                    "component_scores": item.component_scores,
                    "component_ranks": item.component_ranks,
                    "rank": item.rank,
                }
                for item in reranked
            ]

        return [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "text": item.text,
                "metadata": item.metadata,
                "hybrid_score": item.hybrid_score,
                "rerank_score": None,
                "component_scores": item.component_scores,
                "component_ranks": item.component_ranks,
                "rank": idx,
            }
            for idx, item in enumerate(fused_results[:final_k], start=1)
        ]