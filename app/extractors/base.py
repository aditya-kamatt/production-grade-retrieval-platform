from abc import ABC, abstractmethod
from app.ingestion.models import DiscoveredFile, ExtractedDocument

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, discovered_file: DiscoveredFile) -> ExtractedDocument:
        raise NotImplementedError