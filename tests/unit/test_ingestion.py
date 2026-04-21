import numpy as np
import pytest


class FakeChunkRepo:
    def has_document(self, doc_id, content_hash):
        return False

    def upsert_document(self, **kwargs):
        pass

    def replace_chunks(self, **kwargs):
        pass


class FakeEmbeddingRepo:
    def replace_embeddings(self, **kwargs):
        self.called = True


class FakeEmbeddingProvider:
    def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeChunk:
    def __init__(self, cid):
        self.chunk_id = cid
        self.text = "text"
        self.metadata = {}


class FakeExtractor:
    def extract(self, file):
        return file


class FakeNormaliser:
    def normalise(self, doc):
        return doc


class FakeChunker:
    def chunk(self, doc):
        return [FakeChunk("c1"), FakeChunk("c2")]


class FakeFile:
    doc_id = "d1"
    path = "file.txt"
    content_hash = "hash"
    source_path = "file.txt"
    file_type = "txt"
    metadata = {}


class FakeDiscovery:
    def __init__(self, root_dir):
        pass

    def discover(self):
        return [FakeFile()]


def test_ingestion_runs(monkeypatch):
    from app.ingestion.ingestion import IngestionService

    monkeypatch.setattr("app.ingestion.ingestion.FileDiscovery", FakeDiscovery)
    monkeypatch.setattr("app.ingestion.ingestion.DocumentExtractor", FakeExtractor)
    monkeypatch.setattr("app.ingestion.ingestion.DocumentNormaliser", FakeNormaliser)
    monkeypatch.setattr("app.ingestion.ingestion.DocumentChunker", lambda **kwargs: FakeChunker())

    service = IngestionService(
        chunk_repository=FakeChunkRepo(),
        embedding_repository=FakeEmbeddingRepo(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    result = service.ingest_directory("dummy")

    assert result.stats.processed_files == 1
    assert result.stats.total_chunks == 2


def test_write_export_files(tmp_path):
    from app.ingestion.ingestion import IngestionService

    service = IngestionService(chunk_repository=FakeChunkRepo())
    service._processed_output_dir = tmp_path

    service._export_chunks = [{"chunk_id": "c1"}]
    service._export_embeddings = [[0.1, 0.2]]

    service._write_export_files()

    assert (tmp_path / "chunks.json").exists()
    assert (tmp_path / "embeddings.npy").exists()