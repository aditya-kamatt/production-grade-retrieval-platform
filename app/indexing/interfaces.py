from __future__ import annotations

from typing import List, Optional, Protocol

from app.processing.models import Chunk


class ChunkRepository(Protocol):
    def has_document(self, doc_id: str, content_hash: str) -> bool:
        ...

    def upsert_document(
        self,
        doc_id: str,
        source_path: str,
        file_type: str,
        content_hash: str,
        metadata: dict,
    ) -> None:
        ...

    def replace_chunks(self, doc_id: str, chunks: List[Chunk]) -> None:
        ...

    def count_chunks(self) -> int:
        ...

    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict]:
        ...


class EmbeddingRepository(Protocol):
    def replace_embeddings(
        self,
        doc_id: str,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        metadata_list: Optional[List[dict]] = None,
    ) -> None:
        ...

    def search(self, query_vector: List[float], top_k: int) -> List[dict]:
        ...