from pathlib import Path

from reportlab.pdfgen import canvas

from app.extractors.pdf_extractor import PDFExtractor
from app.ingestion.models import DiscoveredFile


def _make_discovered_file(path: Path) -> DiscoveredFile:
    return DiscoveredFile(
        doc_id="doc-pdf",
        path=str(path),
        file_type="pdf",
        size_bytes=path.stat().st_size,
        modified_time=path.stat().st_mtime,
        content_hash="hash-pdf",
    )


def _create_pdf(path: Path, lines: list[str]) -> None:
    pdf = canvas.Canvas(str(path))
    y = 800
    for line in lines:
        pdf.drawString(100, y, line)
        y -= 20
    pdf.save()


def test_pdf_extractor_extracts_text_and_pages(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    _create_pdf(file_path, ["Hello PDF", "Second line"])

    extractor = PDFExtractor()
    doc = extractor.extract(_make_discovered_file(file_path))

    assert "Hello PDF" in doc.raw_text
    assert "Second line" in doc.raw_text
    assert doc.pages is not None
    assert len(doc.pages) == 1
    assert doc.metadata["page_count"] == 1
    assert doc.metadata["extractor"] == "pypdf"