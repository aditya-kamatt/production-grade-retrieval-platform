from pathlib import Path
from app.extractors.base import BaseExtractor
from app.ingestion.models import DiscoveredFile, ExtractedDocument

class TextExtractor(BaseExtractor):
    FALLBACK_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

    def extract(self, discovered_file: DiscoveredFile) -> ExtractedDocument:
        path = Path(discovered_file.path)
        text, encoding_used = self._read_text_with_fallback(path)

        metadata = {
            "source_path": discovered_file.path,
            "file_type": discovered_file.file_type,
            "size_bytes": discovered_file.size_bytes,
            "modified_time": discovered_file.modified_time,
            "content_hash": discovered_file.content_hash,
            "encoding": encoding_used,
            "line_count": text.count("\n") + 1 if text else 0,
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
    
    def _read_text_with_fallback(self, path: Path) -> tuple[str, str]:
        for encoding in self.FALLBACK_ENCODINGS:
            try:
                return path.read_text(encoding=encoding), encoding
            except UnicodeDecodeError:
                continue

        return path.read_text(encoding="utf-8", errors="replace"), "utf-8-replace"