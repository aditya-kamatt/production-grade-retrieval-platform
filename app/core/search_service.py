from __future__ import annotations

from typing import Dict, List
from dataclasses import dataclass
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from app.indexing.faiss_store import FAISSEmbeddingRepository
from app.indexing.sqlite_store import SQLiteChunkRepository
from app.retrieval.lexical_retriever import BM25Retriever, LexicalSearchResult
from app.retrieval.reranker import CrossEncoderReranker, RerankedSearchResult


@dataclass(slots=True)
class HybridCandidate:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict
    hybrid_score: float
    component_scores: dict[str, float]
    component_ranks: dict[str, int]


class SearchService:
    def __init__(
        self,
        sqlite_db_path: str = "data/processed/metadata.db",
        faiss_index_path: str = "data/processed/vector.index",
        faiss_id_map_path: str = "data/processed/vector_ids.json",
        embedding_dimension: int = 384,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:

        self.chunk_repository = SQLiteChunkRepository(sqlite_db_path)
        self.embedding_repository = FAISSEmbeddingRepository(
            index_path=faiss_index_path,
            id_map_path=faiss_id_map_path,
            embedding_dimension=embedding_dimension,
        )

        self.chunks = self.chunk_repository.get_all_chunks()
        if not self.chunks:
            raise ValueError("No chunks found. Run ingestion first.")

        for chunk in self.chunks:
            chunk["document_id"] = chunk["doc_id"]

        self.chunk_by_id: Dict[str, dict] = {
            chunk["chunk_id"]: chunk for chunk in self.chunks
        }

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.lexical_retriever = BM25Retriever(self.chunks)
        self.reranker = CrossEncoderReranker(reranker_model_name)

    def _embed(self, query: str) -> List[float]:
        return self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )[0].astype(np.float32).tolist()

    def _vector_search(self, query: str, top_k: int):
        query_vec = self._embed(query)
        hits = self.embedding_repository.search(query_vec, top_k)

        results = []
        for rank, hit in enumerate(hits, start=1):
            chunk = self.chunk_by_id.get(hit["chunk_id"])
            if not chunk:
                continue

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["doc_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": float(hit["score"]),
                    "rank": rank,
                }
            )
        return results

    def _fuse(
        self,
        lexical: List[LexicalSearchResult],
        vector: List[dict],
        k: int,
    ) -> List[HybridCandidate]:

        fused: Dict[str, HybridCandidate] = {}

        for item in lexical:
            fused[item.chunk_id] = HybridCandidate(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                text=item.text,
                metadata=item.metadata,
                hybrid_score=1 / (60 + item.rank),
                component_scores={"bm25": item.score, "vector": 0.0},
                component_ranks={"bm25": item.rank},
            )

        for item in vector:
            cid = item["chunk_id"]

            if cid not in fused:
                fused[cid] = HybridCandidate(
                    chunk_id=cid,
                    document_id=item["document_id"],
                    text=item["text"],
                    metadata=item["metadata"],
                    hybrid_score=0.0,
                    component_scores={"bm25": 0.0, "vector": item["score"]},
                    component_ranks={"vector": item["rank"]},
                )

            fused[cid].hybrid_score += 1 / (60 + item["rank"])
            fused[cid].component_scores["vector"] = item["score"]
            fused[cid].component_ranks["vector"] = item["rank"]

        results = list(fused.values())
        results.sort(key=lambda x: x.hybrid_score, reverse=True)

        return results[:k]

    def search(
        self,
        query: str,
        candidate_k: int = 20,
        final_k: int = 5,
        use_reranker: bool = True,
    ) -> dict:  

        start_time = time.perf_counter()  

        lexical = self.lexical_retriever.search(query, candidate_k)
        vector = self._vector_search(query, candidate_k)
        fused = self._fuse(lexical, vector, candidate_k)

        if use_reranker:
            reranked: List[RerankedSearchResult] = self.reranker.rerank(
                query=query,
                candidates=fused,
                top_k=final_k,
            )

            results = [
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "text": r.text,
                    "metadata": r.metadata,
                    "rank": rank,  
                    "scores": {
                        "bm25": r.component_scores.get("bm25", 0.0),
                        "vector": r.component_scores.get("vector", 0.0),
                        "hybrid": r.hybrid_score,
                        "reranker": r.rerank_score,
                    },
                }
                for rank, r in enumerate(reranked, start=1)
            ]
        else:
            results = [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "text": c.text,
                    "metadata": c.metadata,
                    "rank": rank,  
                    "scores": {
                        "bm25": c.component_scores.get("bm25", 0.0),
                        "vector": c.component_scores.get("vector", 0.0),
                        "hybrid": c.hybrid_score,
                        "reranker": None,
                    },
                }
                for rank, c in enumerate(fused[:final_k], start=1)
            ]

        latency_ms = (time.perf_counter() - start_time) * 1000  

        return {
            "query": query,
            "latency_ms": round(latency_ms, 3),
            "results": results,
        }