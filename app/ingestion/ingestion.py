from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional

from app.indexing.interfaces import ChunkRepository, EmbeddingRepository
from app.ingestion.discovery import FileDiscovery
from app.extractors.extract import DocumentExtractor
from app.extractors.extract import DocumentExtractor
from app.processing.chunking import DocumentChunker
from app.processing.models import Chunk
from app.processing.normalise import DocumentNormaliser


logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    discovered_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    total_duration_seconds: float = 0.0


@dataclass
class IngestionResult:
    stats: IngestionStats
    failures: List[Dict[str, str]]


class IngestionService:
    def __init__(
        self,
        chunk_repository: ChunkRepository,
        embedding_repository: Optional[EmbeddingRepository] = None,
        embedding_provider: Optional[object] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        embed_batch_size: int = 64,
    ) -> None:
        self.chunk_repository = chunk_repository
        self.embedding_repository = embedding_repository
        self.embedding_provider = embedding_provider
        self.embed_batch_size = embed_batch_size

        self.extractor = DocumentExtractor()
        self.normaliser = DocumentNormaliser()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

    def ingest_directory(self, root_dir: str) -> IngestionResult:
        started_at = perf_counter()
        stats = IngestionStats()
        failures: List[Dict[str, str]] = []

        discovery = FileDiscovery(root_dir=root_dir)
        discovered_files = discovery.discover()
        stats.discovered_files = len(discovered_files)

        logger.info("Discovered %s files in %s", stats.discovered_files, root_dir)

        for discovered_file in discovered_files:
            try:
                if self.chunk_repository.has_document(
                    doc_id=discovered_file.doc_id,
                    content_hash=discovered_file.content_hash,
                ):
                    stats.skipped_files += 1
                    logger.info("Skipping unchanged file: %s", discovered_file.path)
                    continue

                chunk_count = self._ingest_file(discovered_file)
                stats.processed_files += 1
                stats.total_chunks += chunk_count

            except Exception as e:
                stats.failed_files += 1
                logger.exception("Failed to ingest file: %s", discovered_file.path)
                failures.append(
                    {
                        "path": discovered_file.path,
                        "doc_id": discovered_file.doc_id,
                        "error": str(e),
                    }
                )

        stats.total_duration_seconds = perf_counter() - started_at
        return IngestionResult(stats=stats, failures=failures)

    def _ingest_file(self, discovered_file) -> int:
        extracted_document = self.extractor.extract(discovered_file)
        normalised_document = self.normaliser.normalise(extracted_document)
        chunks = self.chunker.chunk(normalised_document)

        self.chunk_repository.upsert_document(
            doc_id=normalised_document.doc_id,
            source_path=normalised_document.source_path,
            file_type=normalised_document.file_type,
            content_hash=discovered_file.content_hash,
            metadata=normalised_document.metadata,
        )

        self.chunk_repository.replace_chunks(
            doc_id=normalised_document.doc_id,
            chunks=chunks,
        )

        logger.info(
            "Stored %s chunks for document %s",
            len(chunks),
            normalised_document.source_path,
        )

        if (
            self.embedding_provider is not None
            and self.embedding_repository is not None
            and chunks
        ):
            self._embed_and_store_chunks(
                doc_id=normalised_document.doc_id,
                chunks=chunks,
            )
        
        return len(chunks)

    def _embed_and_store_chunks(self, doc_id: str, chunks: List[Chunk]) -> None:
        if self.embedding_repository is None:
            logger.warning("Embedding repository not set; skipping storage")
            return
        if self.embedding_provider is None:
            logger.warning("Embedding provider not set; skipping embedding")
            return
        
        chunk_ids: List[str] = []
        vectors: List[List[float]] = []
        metadata_list: List[dict] = []

        for batch_start in range(0, len(chunks), self.embed_batch_size):
            batch = chunks[batch_start : batch_start + self.embed_batch_size]
            batch_texts = [chunk.text for chunk in batch]

            batch_vectors = self.embedding_provider.embed_texts(batch_texts) # type: ignore

            if len(batch_vectors) != len(batch):
                raise ValueError(
                    "Embedding provider returned mismatched number of vectors"
                )

            for chunk, vector in zip(batch, batch_vectors):
                chunk_ids.append(chunk.chunk_id)
                vectors.append(vector)
                metadata_list.append(chunk.metadata)

        self.embedding_repository.replace_embeddings(
            doc_id=doc_id,
            chunk_ids=chunk_ids,
            embeddings=vectors,
            metadata_list=metadata_list,
        )

        logger.info("Stored %s embeddings for document %s", len(chunk_ids), doc_id)

    def _count_total_chunks(self, root_dir: str) -> int:
        if hasattr(self.chunk_repository, 'count_chunks') and self.chunk_repository is not None:
            count = getattr(self.chunk_repository, 'count_chunks', lambda: 0)()
            return count if isinstance(count, int) else 0
        return 0