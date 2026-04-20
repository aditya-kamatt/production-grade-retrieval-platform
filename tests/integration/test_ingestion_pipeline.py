from pathlib import Path

from app.indexing.faiss_store import FAISSEmbeddingRepository
from app.indexing.sqlite_store import SQLiteChunkRepository
from app.ingestion.ingestion import IngestionService
from app.indexing.hybrid_store import HybridRetrievalStore


class FakeEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.lower()
            shipping_score = 1.0 if "shipping" in lower else 0.0
            invoice_score = 1.0 if "invoice" in lower else 0.0
            generic_score = float(len(text) % 10) / 10.0
            vectors.append([shipping_score, invoice_score, generic_score])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        lower = text.lower()
        shipping_score = 1.0 if "shipping" in lower else 0.0
        invoice_score = 1.0 if "invoice" in lower else 0.0
        generic_score = float(len(text) % 10) / 10.0
        return [shipping_score, invoice_score, generic_score]


def test_ingestion_pipeline_end_to_end(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    (raw_dir / "shipping.txt").write_text(
        "Shipping notice for order 123. Port of export is Felixstowe.",
        encoding="utf-8",
    )
    (raw_dir / "invoice.txt").write_text(
        "Invoice 456. Payment due in 30 days.",
        encoding="utf-8",
    )
    (raw_dir / "data.csv").write_text(
        "customer,status\nAcme,shipped\nGlobex,pending\n",
        encoding="utf-8",
    )

    chunk_repository = SQLiteChunkRepository(str(tmp_path / "metadata.db"))
    embedding_repository = FAISSEmbeddingRepository(
        index_path=str(tmp_path / "vector.index"),
        id_map_path=str(tmp_path / "vector_ids.json"),
        embedding_dimension=3,
    )
    embedding_provider = FakeEmbeddingProvider()

    service = IngestionService(
        chunk_repository=chunk_repository,
        embedding_repository=embedding_repository,
        embedding_provider=embedding_provider,
        chunk_size=120,
        chunk_overlap=20,
        min_chunk_size=10,
        embed_batch_size=8,
    )

    result = service.ingest_directory(str(raw_dir))

    assert result.stats.discovered_files == 3
    assert result.stats.processed_files == 3
    assert result.stats.failed_files == 0
    assert result.stats.total_chunks > 0
    assert chunk_repository.count_chunks() > 0
    assert embedding_repository.index.ntotal > 0

    retrieval_store = HybridRetrievalStore(
        chunk_repository=chunk_repository,
        embedding_repository=embedding_repository,
    )

    query_vector = embedding_provider.embed_query("shipping")
    results = retrieval_store.semantic_search(query_vector=query_vector, top_k=3)

    assert len(results) > 0
    assert any("Shipping" in result["text"] or "shipped" in result["text"] for result in results)


def test_ingestion_skips_unchanged_documents_on_rerun(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    (raw_dir / "sample.txt").write_text("Hello world", encoding="utf-8")

    chunk_repository = SQLiteChunkRepository(str(tmp_path / "metadata.db"))
    embedding_repository = FAISSEmbeddingRepository(
        index_path=str(tmp_path / "vector.index"),
        id_map_path=str(tmp_path / "vector_ids.json"),
        embedding_dimension=3,
    )
    embedding_provider = FakeEmbeddingProvider()

    service = IngestionService(
        chunk_repository=chunk_repository,
        embedding_repository=embedding_repository,
        embedding_provider=embedding_provider,
        chunk_size=100,
        chunk_overlap=10,
        min_chunk_size=5,
        embed_batch_size=8,
    )

    first_run = service.ingest_directory(str(raw_dir))
    second_run = service.ingest_directory(str(raw_dir))

    assert first_run.stats.processed_files == 1
    assert second_run.stats.skipped_files == 1
    assert second_run.stats.processed_files == 0