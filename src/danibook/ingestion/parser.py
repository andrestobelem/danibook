from pathlib import Path
from typing import List, Tuple, Optional, Dict
import hashlib

from .extractors import (
    DocumentChunk,
    ParsedDocument,
    BaseExtractor,
    TextExtractor,
    DocumentExtractor,
    MediaExtractor,
    WebIngestor,
)


class DocumentParser:
    """Facade for document parsing and chunking across diverse local source formats."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg"}
    TABLE_EXTENSIONS = {".csv", ".tsv"}
    TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".json"}
    DOC_EXTENSIONS = {".pdf", ".docx", ".epub", ".html"}

    SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | TABLE_EXTENSIONS | TEXT_EXTENSIONS | DOC_EXTENSIONS

    _text_extractor = TextExtractor()
    _doc_extractor = DocumentExtractor()
    _media_extractor = MediaExtractor()

    @staticmethod
    def calculate_file_hash(path: Path) -> str:
        """Calculates a SHA-256 hash of the file contents for incremental change detection."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def parse(cls, path: Path) -> ParsedDocument:
        """Parses any supported source into a canonical ParsedDocument."""
        fhash = cls.calculate_file_hash(path)
        ext = path.suffix.lower()

        if ext in cls.TABLE_EXTENSIONS:
            media_type = "table"
            chunks = cls._text_extractor.extract_chunks(path)
            image_data_uri = None
        elif ext in cls.IMAGE_EXTENSIONS:
            media_type = "image"
            chunks = cls._media_extractor.extract_chunks(path)
            image_data_uri = cls._media_extractor.extract_image_data_uri(path)
        elif ext in cls.AUDIO_EXTENSIONS:
            media_type = "audio"
            chunks = cls._media_extractor.extract_chunks(path)
            image_data_uri = None
        elif ext in cls.DOC_EXTENSIONS:
            media_type = "text"
            chunks = cls._doc_extractor.extract_chunks(path)
            image_data_uri = None
        else:
            media_type = "text"
            chunks = cls._text_extractor.extract_chunks(path)
            image_data_uri = None

        full_text = "\n\n".join(c.content for c in chunks)
        return ParsedDocument(
            source_file=path.name,
            file_path=path,
            file_hash=fhash,
            chunks=chunks,
            full_text=full_text,
            media_type=media_type,
            image_data_uri=image_data_uri,
        )

    @classmethod
    def ingest_url_to_file(cls, url: str, workspace_dir: Path) -> Path:
        """Fetches a URL or YouTube transcript and stores it as a Markdown source."""
        return WebIngestor.ingest_url_to_file(url, workspace_dir)
