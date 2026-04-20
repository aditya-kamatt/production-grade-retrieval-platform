from pathlib import Path

from app.extractors.csv_extractor import CSVExtractor
from app.ingestion.models import DiscoveredFile


def _make_discovered_file(path: Path) -> DiscoveredFile:
    return DiscoveredFile(
        doc_id="doc-csv",
        path=str(path),
        file_type="csv",
        size_bytes=path.stat().st_size,
        modified_time=path.stat().st_mtime,
        content_hash="hash-csv",
    )


def test_csv_extractor_converts_rows_to_searchable_text(tmp_path: Path) -> None:
    file_path = tmp_path / "table.csv"
    file_path.write_text(
        "customer,country,status\nAcme,UK,shipped\nGlobex,US,pending\n",
        encoding="utf-8",
    )

    extractor = CSVExtractor()
    doc = extractor.extract(_make_discovered_file(file_path))

    assert "CSV DOCUMENT" in doc.raw_text
    assert "COLUMNS: customer, country, status" in doc.raw_text
    assert "ROW 1: customer=Acme | country=UK | status=shipped" in doc.raw_text
    assert doc.metadata["row_count"] == 2
    assert doc.metadata["column_count"] == 3


def test_csv_extractor_handles_missing_values(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.csv"
    file_path.write_text("a,b,c\n1,2\n", encoding="utf-8")

    extractor = CSVExtractor()
    doc = extractor.extract(_make_discovered_file(file_path))

    assert "ROW 1: a=1 | b=2 | c=" in doc.raw_text
    assert doc.metadata["row_count"] == 1


def test_csv_extractor_handles_empty_file(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.csv"
    file_path.write_text("", encoding="utf-8")

    extractor = CSVExtractor()
    doc = extractor.extract(_make_discovered_file(file_path))

    assert doc.raw_text == ""
    assert doc.metadata["row_count"] == 0
    assert doc.metadata["column_count"] == 0