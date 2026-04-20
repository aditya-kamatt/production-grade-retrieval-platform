from app.processing.chunking import DocumentChunker
from app.processing.models import NormalisedDocument


def test_chunker_returns_empty_list_for_blank_text() -> None:
    doc = NormalisedDocument(
        doc_id="doc-1",
        source_path="x",
        file_type="txt",
        normalised_text="   ",
        metadata={},
        pages=None,
    )

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
    chunks = chunker.chunk(doc)

    assert chunks == []


def test_chunker_creates_multiple_chunks_with_overlap() -> None:
    text = "A" * 300 + "\n\n" + "B" * 300 + "\n\n" + "C" * 300
    doc = NormalisedDocument(
        doc_id="doc-2",
        source_path="x",
        file_type="txt",
        normalised_text=text,
        metadata={},
        pages=None,
    )

    chunker = DocumentChunker(chunk_size=250, chunk_overlap=50, min_chunk_size=20)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3

    for index, chunk in enumerate(chunks):
        assert chunk.doc_id == "doc-2"
        assert chunk.chunk_index == index
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["start_char"] < chunk.metadata["end_char"]
        assert len(chunk.text) > 0


def test_chunker_rejects_invalid_overlap() -> None:
    try:
        DocumentChunker(chunk_size=100, chunk_overlap=100, min_chunk_size=10)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "chunk_overlap must be smaller than chunk_size" in str(exc)


def test_chunk_ids_are_unique() -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three." * 20
    doc = NormalisedDocument(
        doc_id="doc-3",
        source_path="x",
        file_type="txt",
        normalised_text=text,
        metadata={},
        pages=None,
    )

    chunker = DocumentChunker(chunk_size=120, chunk_overlap=20, min_chunk_size=10)
    chunks = chunker.chunk(doc)

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))