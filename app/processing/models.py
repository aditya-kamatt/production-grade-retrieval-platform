from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.ingestion.models import PageContent


@dataclass
class NormalisedDocument:
    doc_id: str
    source_path: str
    file_type: str
    normalised_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: Optional[List[PageContent]] = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)