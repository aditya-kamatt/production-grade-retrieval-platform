from pathlib import Path

from app.extractors.text_extractor import TextExtractor
from app.ingestion.models import DiscoveredFile


def _make_discovered_file(path: Path, file_type: str = "txt") -> DiscoveredFile:
    return DiscoveredFile(
        doc_id="doc-1",
        path=str(path),
        file_type=file_type,
        size_bytes=path.stat().st_size,
        modified_time=path.stat().st_mtime,
        content_hash="hash-1",
    )


def test_text_extractor_reads_utf8_file(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello\nworld", encoding="utf-8")

    extractor = TextExtractor()
    doc = extractor.extract(_make_discovered_file(file_path))

    assert doc.raw_text == "hello\nworld"
    assert doc.file_type == "txt"
    assert doc.metadata["char_count"] == len("hello\nworld")
    assert doc.metadata["line_count"] == 2


def test_text_extractor_reads_markdown_file(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Title\n\nSome text", encoding="utf-8")

    extractor = TextExtractor()
    doc = extractor.extract(_make_discovered_file(file_path, file_type="md"))

    assert "# Title" in doc.raw_text
    assert doc.file_type == "md"


def test_text_extractor_falls_back_on_non_utf8(tmp_path: Path) -> None:
    file_path = tmp_path / "latin1.txt"
    file_path.write_bytes("café".encode("latin-1"))

    extractor = TextExtractor()
    doc = extractor.extract(_make_discovered_file(file_path))

    assert "café" in doc.raw_text
    assert doc.metadata["encoding"] in {"latin-1", "cp1252", "utf-8-replace"}