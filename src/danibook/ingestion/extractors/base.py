from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import hashlib


@dataclass
class DocumentChunk:
    source_file: str
    content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def text(self) -> str:
        """Alias preserving backward compatibility."""
        return self.content


@dataclass
class ParsedDocument:
    source_file: str
    file_path: Path
    file_hash: str
    chunks: List[DocumentChunk]
    full_text: str
    media_type: str = "text"  # "text", "image", "audio", "web", "youtube", "table"
    image_data_uri: Optional[str] = None


class BaseExtractor(ABC):
    """Abstract base class for modular source extraction."""

    @abstractmethod
    def extract_chunks(self, path: Path) -> List[DocumentChunk]:
        """Extracts structured DocumentChunk items from the given file."""
        pass
