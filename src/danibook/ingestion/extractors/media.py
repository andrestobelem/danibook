from pathlib import Path
from typing import List, Tuple
import base64
from .base import BaseExtractor, DocumentChunk


class MediaExtractor(BaseExtractor):
    """Extracts chunks and data URIs from images and audio files."""

    def extract_chunks(self, path: Path) -> List[DocumentChunk]:
        ext = path.suffix.lower()
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            return self._extract_image_chunks(path)
        else:
            return self._extract_audio_chunks(path)

    def extract_image_data_uri(self, path: Path) -> str:
        data = path.read_bytes()
        ext = path.suffix.lower().replace(".", "")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/{mime};base64,{b64}"

    def _extract_image_chunks(self, path: Path) -> List[DocumentChunk]:
        size_kb = path.stat().st_size // 1024 if path.exists() else 0
        desc = (
            f"[IMAGEN ADJUNTA: {path.name} (~{size_kb} KB)]. "
            f"La imagen está disponible para análisis visual multimodal con Gemma 4."
        )
        return [
            DocumentChunk(
                source_file=path.name,
                content=desc,
                section_title="Adjunto Visual",
            )
        ]

    def _extract_audio_chunks(self, path: Path) -> List[DocumentChunk]:
        size_kb = path.stat().st_size // 1024 if path.exists() else 0
        desc = (
            f"[ARCHIVO DE AUDIO: {path.name} (~{size_kb} KB)]. "
            f"El archivo está registrado en el Cuaderno para análisis auditivo o transcripción multimodal con Gemma 4."
        )
        return [
            DocumentChunk(
                source_file=path.name,
                content=desc,
                section_title="Archivo de Audio",
            )
        ]
