from __future__ import annotations

from dataclasses import asdict

from app.retrieval.hybrid_retriever import reciprocal_rank_fusion
from app.retrieval.lexical_retriever import BM25Retriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import FaissVectorRetriever


class HybridSearchService:
    """
    Full retrieval pipeline:
    lexical -> dense -> fusion -> rerank

    Why this design:
    - Keeps pipeline composable and testable.
    - Lets evaluation compare:
      * BM25 only
      * Dense only
      * Hybrid fused
      * Hybrid + reranker
    """

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
        fusion_weights: dict[str, float] | None = None,
        use_reranker: bool = True,
    ) -> list[dict]:
        lexical_results = self.lexical_retriever.search(query=query, top_k=candidate_k)
        dense_results = self.vector_retriever.search(query=query, top_k=candidate_k)

        fused_results = reciprocal_rank_fusion(
            result_sets={
                "bm25": lexical_results,
                "dense": dense_results,
            },
            top_k=candidate_k,
            weights=fusion_weights or {"bm25": 1.0, "dense": 1.0},
        )

        if use_reranker and self.reranker is not None:
            reranked = self.reranker.rerank(
                query=query,
                candidates=fused_results,
                top_k=final_k,
            )
            return [asdict(item) for item in reranked]

        return [asdict(item) for item in fused_results[:final_k]]