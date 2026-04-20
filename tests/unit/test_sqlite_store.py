from pathlib import Path

from app.indexing.sqlite_store import SQLiteChunkRepository
from app.processing.models import Chunk


def test_sqlite_store_upserts_documents_and_chunks(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    repo = SQLiteChunkRepository(str(db_path))

    repo.upsert_document(
        doc_id="doc-1",
        source_path="data/raw/a.txt",
        file_type="txt",
        content_hash="hash-1",
        metadata={"source": "test"},
    )

    chunks = [
        Chunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            chunk_index=0,
            text="first chunk",
            metadata={"chunk_index": 0},
        ),
        Chunk(
            chunk_id="chunk-2",
            doc_id="doc-1",
            chunk_index=1,
            text="second chunk",
            metadata={"chunk_index": 1},
        ),
    ]

    repo.replace_chunks("doc-1", chunks)

    assert repo.has_document("doc-1", "hash-1") is True
    assert repo.count_chunks() == 2

    chunk = repo.get_chunk_by_id("chunk-1")
    assert chunk is not None
    assert chunk["text"] == "first chunk"
    assert chunk["doc_id"] == "doc-1"


def test_sqlite_store_replaces_chunks_for_same_document(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    repo = SQLiteChunkRepository(str(db_path))

    repo.upsert_document(
        doc_id="doc-1",
        source_path="data/raw/a.txt",
        file_type="txt",
        content_hash="hash-1",
        metadata={},
    )

    repo.replace_chunks(
        "doc-1",
        [
            Chunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                chunk_index=0,
                text="old",
                metadata={},
            )
        ],
    )

    repo.replace_chunks(
        "doc-1",
        [
            Chunk(
                chunk_id="chunk-2",
                doc_id="doc-1",
                chunk_index=0,
                text="new",
                metadata={},
            )
        ],
    )

    assert repo.count_chunks() == 1
    assert repo.get_chunk_by_id("chunk-1") is None

    chunk = repo.get_chunk_by_id("chunk-2")
    assert chunk is not None
    assert chunk["text"] == "new"