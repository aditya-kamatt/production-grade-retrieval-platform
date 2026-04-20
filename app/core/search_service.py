from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.retrieval.hybrid_search import HybridSearchService
from app.retrieval.lexical_retriever import BM25Retriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import FaissVectorRetriever


class SearchService:
    """
    Loads persisted chunks + embeddings and exposes one .search() method
    for the API route to call.
    """

    def __init__(
        self,
        chunks_path: str = "data/processed/chunks.json",
        embeddings_path: str = "data/processed/embeddings.npy",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.embeddings_path = Path(embeddings_path)

        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")

        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")

        with self.chunks_path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)

        embeddings = np.load(self.embeddings_path).astype("float32")

        self.embedding_model = SentenceTransformer(embedding_model_name)

        lexical_retriever = BM25Retriever(chunks=chunks)

        vector_retriever = FaissVectorRetriever(
            chunks=chunks,
            embeddings=embeddings,
            embedding_function=self._embed_texts,
        )

        reranker = CrossEncoderReranker(model_name=reranker_model_name)

        self.hybrid_search = HybridSearchService(
            lexical_retriever=lexical_retriever,
            vector_retriever=vector_retriever,
            reranker=reranker,
        )

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def search(
        self,
        query: str,
        candidate_k: int = 20,
        final_k: int = 5,
        use_reranker: bool = True,
    ) -> list[dict]:
        return self.hybrid_search.search(
            query=query,
            candidate_k=candidate_k,
            final_k=final_k,
            use_reranker=use_reranker,
        )