import logging

from app.core.config import settings
from app.indexing.faiss_store import FAISSEmbeddingRepository
from app.indexing.sqlite_store import SQLiteChunkRepository
from app.ingestion.ingestion import IngestionService
from app.services.embedding_provider import SentenceTransformerEmbeddingProvider


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    chunk_repository = SQLiteChunkRepository(str(settings.sqlite_db_path))
    embedding_repository = FAISSEmbeddingRepository(
        index_path=str(settings.faiss_index_path),
        id_map_path=str(settings.faiss_id_map_path),
        embedding_dimension=settings.embedding_dimension,
    )
    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model_name
    )

    service = IngestionService(
        chunk_repository=chunk_repository,
        embedding_repository=embedding_repository,
        embedding_provider=embedding_provider,
        chunk_size=1200,
        chunk_overlap=200,
        min_chunk_size=100,
        embed_batch_size=64,
    )

    result = service.ingest_directory("data/raw")

    print("Ingestion complete")
    print(result.stats)
    if result.failures:
        print("Failures:")
        for failure in result.failures:
            print(failure)


if __name__ == "__main__":
    main()