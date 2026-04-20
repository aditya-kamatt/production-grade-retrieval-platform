from app.extractors import CSVExtractor, PDFExtractor, TextExtractor
from app.ingestion.models import DiscoveredFile, ExtractedDocument


class UnsupportedFileTypeError(ValueError):
    pass


class DocumentExtractor:
    def __init__(self) -> None:
        self._extractors = {
            "txt": TextExtractor(),
            "md": TextExtractor(),
            "csv": CSVExtractor(),
            "pdf": PDFExtractor(),
        }

    def extract(self, discovered_file: DiscoveredFile) -> ExtractedDocument:
        extractor = self._extractors.get(discovered_file.file_type.lower())

        if extractor is None:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {discovered_file.file_type}"
            )

        return extractor.extract(discovered_file)