from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from app.indexing.interfaces import EmbeddingRepository


class FAISSEmbeddingRepository(EmbeddingRepository):
    def __init__(
        self,
        index_path: str,
        id_map_path: str,
        embedding_dimension: int,
    ) -> None:
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.embedding_dimension = embedding_dimension

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.id_map_path.parent.mkdir(parents=True, exist_ok=True)

        self.index = self._load_or_create_index()
        self.chunk_ids = self._load_chunk_ids()

    def _load_or_create_index(self) -> faiss.Index:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))

        index = faiss.IndexFlatIP(self.embedding_dimension)
        faiss.write_index(index, str(self.index_path))
        return index

    def _load_chunk_ids(self) -> List[str]:
        if self.id_map_path.exists():
            return json.loads(self.id_map_path.read_text(encoding="utf-8"))
        return []

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.id_map_path.write_text(
            json.dumps(self.chunk_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def replace_embeddings(
        self,
        doc_id: str,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        metadata_list: Optional[List[dict]] = None,
    ) -> None:
        """
        FAISS flat indexes do not support easy in-place delete by doc_id.
        So we rebuild the index excluding old chunk_ids for this doc if needed.
        This is acceptable for v1 and small-medium local corpora.
        """
        del metadata_list

        existing_vectors = self._dump_all_vectors()
        existing_pairs = [
            (chunk_id, vector)
            for chunk_id, vector in zip(self.chunk_ids, existing_vectors)
            if not chunk_id.startswith(f"{doc_id}_")
        ]

        new_pairs = list(zip(chunk_ids, embeddings))
        all_pairs = existing_pairs + new_pairs

        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.chunk_ids = []

        if all_pairs:
            vectors = np.array([pair[1] for pair in all_pairs], dtype=np.float32)
            vectors = self._normalize(vectors)
            self.index.add(vectors) # type: ignore
            self.chunk_ids = [pair[0] for pair in all_pairs]

        self._save()

    def search(self, query_vector: List[float], top_k: int) -> List[dict]:
        if self.index.ntotal == 0:
            return []

        query = np.array([query_vector], dtype=np.float32)
        query = self._normalize(query)

        scores, indices = self.index.search(query, top_k) # type: ignore

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            results.append(
                {
                    "chunk_id": self.chunk_ids[index],
                    "score": float(score),
                }
            )
        return results

    def _dump_all_vectors(self) -> np.ndarray:
        if self.index.ntotal == 0:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        vectors = np.zeros((self.index.ntotal, self.embedding_dimension), dtype=np.float32)
        for i in range(self.index.ntotal):
            self.index.reconstruct(i, vectors[i])
        return vectors

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        faiss.normalize_L2(vectors)
        return vectors