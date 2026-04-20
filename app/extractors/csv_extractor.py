import csv
from pathlib import Path
from typing import List

from app.extractors.base import BaseExtractor
from app.ingestion.models import DiscoveredFile, ExtractedDocument


class CSVExtractor(BaseExtractor):
    FALLBACK_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

    def extract(self, discovered_file: DiscoveredFile) -> ExtractedDocument:
        path = Path(discovered_file.path)
        rows, encoding_used = self._read_csv(path)
        text = self._rows_to_text(rows)

        header = rows[0] if rows else []
        body = rows[1:] if len(rows) > 1 else []

        metadata = {
            "source_path": discovered_file.path,
            "file_type": discovered_file.file_type,
            "size_bytes": discovered_file.size_bytes,
            "modified_time": discovered_file.modified_time,
            "content_hash": discovered_file.content_hash,
            "encoding": encoding_used,
            "row_count": len(body),
            "column_count": len(header),
            "char_count": len(text),
        }

        return ExtractedDocument(
            doc_id=discovered_file.doc_id,
            source_path=discovered_file.path,
            file_type=discovered_file.file_type,
            raw_text=text,
            metadata=metadata,
            pages=None,
        )

    def _read_csv(self, path: Path) -> tuple[List[List[str]], str]:
        for encoding in self.FALLBACK_ENCODINGS:
            try:
                with path.open("r", encoding=encoding, newline="") as file:
                    return [row for row in csv.reader(file)], encoding
            except UnicodeDecodeError:
                continue

        with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
            return [row for row in csv.reader(file)], "utf-8-replace"

    def _rows_to_text(self, rows: List[List[str]]) -> str:
        if not rows:
            return ""

        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []

        lines: List[str] = []
        lines.append("CSV DOCUMENT")
        lines.append(f"COLUMNS: {', '.join(self._clean_cell(col) for col in header)}")
        lines.append("")

        for row_index, row in enumerate(body, start=1):
            pairs = []
            for col_index, column_name in enumerate(header):
                value = row[col_index] if col_index < len(row) else ""
                pairs.append(f"{self._clean_cell(column_name)}={self._clean_cell(value)}")

            lines.append(f"ROW {row_index}: " + " | ".join(pairs))

        return "\n".join(lines)

    def _clean_cell(self, value: str) -> str:
        if value is None:
            return ""
        return str(value).replace("\n", " ").replace("\r", " ").strip()