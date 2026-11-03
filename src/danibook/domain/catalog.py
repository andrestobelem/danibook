from pathlib import Path
from typing import Dict, List, Set
from ..config import COLLECTIONS_DIR, SAMPLES_DIR
from ..ingestion.parser import DocumentParser


class CatalogRepository:
    """Provides discovery and file resolution for Collections and Samples."""

    COLLECTION_METADATA: Dict[str, Dict[str, str]] = {
        "cervantes": {
            "name": "Miguel de Cervantes (Siglo de Oro)",
            "description": "Don Quijote de la Mancha completo y Novelas Ejemplares.",
            "icon": "auto_stories",
        },
        "clasicos_universales": {
            "name": "Grandes Clásicos Universales",
            "description": "Guerra y Paz, El Conde de Montecristo, Frankenstein y Alicia en el País de las Maravillas.",
            "icon": "menu_book",
        },
        "ciencia_ia": {
            "name": "Ciencia, IA & Transformers",
            "description": "Paper original de Attention Is All You Need y notas sobre LLMs.",
            "icon": "psychology",
        },
        "psicoanalisis_filosofia": {
            "name": "Psicoanálisis & Ensayo (Freud)",
            "description": "Sigmund Freud Tomo 18 (Más allá del principio del placer) y síntesis clínica.",
            "icon": "neurology",
        },
        "ia_local_edge": {
            "name": "IA Local, Cuantización & Edge",
            "description": "Arquitectura GGUF, Metal, aceleración por hardware y Gemma 4 Multimodal.",
            "icon": "memory",
        },
    }

    def __init__(self, collections_dir: Path = COLLECTIONS_DIR, samples_dir: Path = SAMPLES_DIR):
        self.collections_dir = Path(collections_dir).resolve()
        self.samples_dir = Path(samples_dir).resolve()

    def list_collections(self) -> List[Dict]:
        """Lists all curated thematic Collections and their bundled Source filenames."""
        if not self.collections_dir.exists():
            return []
        result = []
        for col_dir in sorted(self.collections_dir.iterdir()):
            if col_dir.is_dir() and not col_dir.name.startswith("."):
                cid = col_dir.name
                files = [f.name for f in sorted(col_dir.iterdir()) if f.is_file() and not f.name.startswith(".")]
                meta = self.COLLECTION_METADATA.get(cid, {
                    "name": cid.replace("_", " ").title(),
                    "description": f"Colección con {len(files)} documentos preparados.",
                    "icon": "collections_bookmark",
                })
                result.append({
                    "id": cid,
                    "name": meta["name"],
                    "description": meta["description"],
                    "icon": meta["icon"],
                    "files_count": len(files),
                    "files": files,
                })
        return result

    def resolve_collection_files(self, collection_id: str) -> List[Path]:
        """Returns the list of file paths inside a Collection."""
        target_dir = (self.collections_dir / collection_id).resolve()
        if target_dir.parent != self.collections_dir or not target_dir.exists() or not target_dir.is_dir():
            raise ValueError(f"Collection not found: {collection_id}")
        return [f for f in sorted(target_dir.iterdir()) if f.is_file() and not f.name.startswith(".")]

    def list_samples(self, active_source_names: Set[str]) -> List[Dict]:
        """Lists all standalone Sample files available in the catalog."""
        samples = []
        if not self.samples_dir.exists():
            return samples
        for p in sorted(self.samples_dir.iterdir()):
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in DocumentParser.SUPPORTED_EXTENSIONS:
                size_kb = p.stat().st_size / 1024.0
                size_str = f"{size_kb / 1024.0:.2f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
                samples.append({
                    "filename": p.name,
                    "size": size_str,
                    "ext": p.suffix.lower(),
                    "is_in_workspace": p.name in active_source_names,
                })
        return samples

    def resolve_sample_files(self, filenames: List[str]) -> List[Path]:
        """Resolves requested Sample filenames into verified paths."""
        resolved = []
        for name in filenames:
            src = (self.samples_dir / Path(name).name).resolve()
            if src.parent == self.samples_dir and src.exists() and src.is_file():
                resolved.append(src)
        return resolved
