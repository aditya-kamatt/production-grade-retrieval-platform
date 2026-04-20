from app.ingestion.models import ExtractedDocument
from app.processing.normalise import DocumentNormaliser


def test_normaliser_cleans_text_but_preserves_structure() -> None:
    document = ExtractedDocument(
        doc_id="doc-1",
        source_path="data/raw/sample.txt",
        file_type="txt",
        raw_text="hello\r\n\r\n\r\nworld\x00   \nline\t\twith   spaces",
        metadata={"k": "v"},
        pages=None,
    )

    normaliser = DocumentNormaliser()
    normalised = normaliser.normalise(document)

    assert "\r" not in normalised.normalised_text
    assert "\x00" not in normalised.normalised_text
    assert "\n\n\n" not in normalised.normalised_text
    assert "line with spaces" in normalised.normalised_text
    assert normalised.metadata["normalised_char_count"] == len(normalised.normalised_text)
    assert normalised.doc_id == document.doc_id


def test_normalizer_keeps_page_markers() -> None:
    document = ExtractedDocument(
        doc_id="doc-2",
        source_path="data/raw/file.pdf",
        file_type="pdf",
        raw_text="--- PAGE 1 START ---\nSome text\n--- PAGE 1 END ---",
        metadata={},
        pages=None,
    )

    normaliser = DocumentNormaliser()
    normalised = normaliser.normalise(document)

    assert "--- PAGE 1 START ---" in normalised.normalised_text
    assert "--- PAGE 1 END ---" in normalised.normalised_text