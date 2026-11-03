from typing import Dict, List, Optional
from pathlib import Path
from ..ingestion.parser import ParsedDocument

class DeskMemory:
    """Tier 1: Mesa de Trabajo (Active Chapters & Core Notes).
    
    Manages active documents kept in continuous prefix context.
    Builds the structured prefix that gets cached on disk for near-instant inference.
    """

    def __init__(self):
        self.active_docs: Dict[str, ParsedDocument] = {}

    def add_document(self, doc: ParsedDocument):
        self.active_docs[doc.source_file] = doc

    def remove_document(self, source_file: str):
        if source_file in self.active_docs:
            del self.active_docs[source_file]

    def clear(self):
        self.active_docs.clear()

    def get_active_files(self) -> List[str]:
        return list(self.active_docs.keys())

    def get_active_image_uris(self, max_images: int = 3) -> List[str]:
        """Returns base64 data URIs of active image sources for Gemma 4 multimodal vision."""
        uris = []
        for filename in sorted(self.active_docs.keys()):
            doc = self.active_docs[filename]
            if doc.media_type == "image" and doc.image_data_uri:
                uris.append(doc.image_data_uri)
                if len(uris) >= max_images:
                    break
        return uris

    def build_desk_prefix(self, max_chars: int = 16000) -> str:
        """Constructs an append-only, ordered prefix string for the LLM prompt.
        
        Protects the model context window by budgeting characters across active desk documents.
        """
        if not self.active_docs:
            return ""

        parts = ["# MESA DE TRABAJO (Capítulos y Borradores Activos)\n"]
        budget_per_doc = max(300, max_chars // max(1, len(self.active_docs)))

        # Sort keys to preserve deterministic token ordering for disk cache hit
        for filename in sorted(self.active_docs.keys()):
            if sum(len(p) for p in parts) >= max_chars:
                parts.append(f"\n[... {len(self.active_docs)} fuentes activas indexadas en búsqueda BM25 ...]")
                break
            doc = self.active_docs[filename]
            parts.append(f"## Documento Activo: {filename}")
            text = "".join(ch for ch in doc.full_text.strip() if ch == "\n" or ch == "\t" or ord(ch) >= 32)
            if len(text) > budget_per_doc:
                text = text[:budget_per_doc] + f"\n\n[... Fragmento de {budget_per_doc} caracteres mostrado de un total de {len(doc.full_text):,}. El resto del libro está disponible para búsqueda en Archivo ...]"
            parts.append(text)
            parts.append("\n---\n")

        return "\n\n".join(parts)
