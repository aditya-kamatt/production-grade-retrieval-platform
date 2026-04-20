from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass 
class DiscoveredFile:
    doc_id: str 
    path: str 
    file_type: str 
    size_bytes: int 
    modified_time: float
    content_hash: str 

@dataclass
class PageContent:
    page_num: int
    text: str

@dataclass
class ExtractedDocument:
    doc_id: str
    source_path: str
    file_type: str
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: Optional[List[PageContent]] = None