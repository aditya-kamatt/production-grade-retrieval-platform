from pathlib import Path
from typing import List

from pypdf import PdfReader

from app.extractors.base import BaseExtractor
from app.ingestion.models import DiscoveredFile, ExtractedDocument, PageContent


class PDFExtractor(BaseExtractor):
    def extract(self, discovered_file: DiscoveredFile) -> ExtractedDocument:
        path = Path(discovered_file.path)
        reader = PdfReader(str(path))

        pages: List[PageContent] = []

        for index, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            pages.append(PageContent(page_num=index, text=page_text))

        raw_text = self._merge_pages(pages)

        metadata = {
            "source_path": discovered_file.path,
            "file_type": discovered_file.file_type,
            "size_bytes": discovered_file.size_bytes,
            "modified_time": discovered_file.modified_time,
            "content_hash": discovered_file.content_hash,
            "page_count": len(pages),
            "char_count": len(raw_text),
            "extractor": "pypdf",
        }

        return ExtractedDocument(
            doc_id=discovered_file.doc_id,
            source_path=discovered_file.path,
            file_type=discovered_file.file_type,
            raw_text=raw_text,
            metadata=metadata,
            pages=pages,
        )

    def _merge_pages(self, pages: List[PageContent]) -> str:
        merged: List[str] = []

        for page in pages:
            merged.append(f"\n--- PAGE {page.page_num} START ---\n")
            merged.append(page.text)
            merged.append(f"\n--- PAGE {page.page_num} END ---\n")

        return "".join(merged).strip()