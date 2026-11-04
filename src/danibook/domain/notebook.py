from pathlib import Path
from typing import Dict, Iterator, List, Optional
import re
import shutil

from ..config import WORKSPACE_DIR, NOTES_DIR, COLLECTIONS_DIR, SAMPLES_DIR, LOCAL_GGUF_PATH, DEFAULT_MODEL_ID
from ..ingestion.parser import DocumentParser, ParsedDocument
from ..ingestion.watcher import FolderWatcher
from ..ingestion.researcher import DeepResearcher
from ..memory.tier1_desk import DeskMemory
from ..memory.tier2_archive import ArchiveMemory
from ..llm.engine import LLMEngine
from .studio import StudioStore
from .catalog import CatalogRepository
from .notebook_entity import Notebook
from .synthesizer import StudioSynthesizer


class NotebookService:
    """Domain service coordinating Sources, Desk (Tier 1), Archive (Tier 2), Studio Notes, Catalog, and LLM synthesis."""

    def __init__(
        self,
        workspace_dir: Path = WORKSPACE_DIR,
        notes_dir: Path = NOTES_DIR,
        collections_dir: Path = COLLECTIONS_DIR,
        samples_dir: Path = SAMPLES_DIR,
        engine: Optional[LLMEngine] = None,
        desk: Optional[DeskMemory] = None,
        archive: Optional[ArchiveMemory] = None,
        active_notebook: Optional[Notebook] = None,
    ):
        self.active_notebook = active_notebook or Notebook(
            id="default",
            title="Cuaderno Principal",
        )
        # If default dirs are passed, bind them to workspace_dir and notes_dir
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.watcher = FolderWatcher(self.workspace_dir)
        self.desk = desk or DeskMemory()
        self.archive = archive or ArchiveMemory()
        self.studio = StudioStore(notes_dir)
        self.catalog = CatalogRepository(collections_dir, samples_dir)
        self.researcher = DeepResearcher(max_results=5)
        self.engine = engine or LLMEngine(
            model_path=LOCAL_GGUF_PATH,
            use_mock=not LOCAL_GGUF_PATH.exists(),
            model_id=DEFAULT_MODEL_ID,
        )
        self.sources: Dict[str, ParsedDocument] = {}
        self.initial_sync()

    @property
    def notebook_id(self) -> str:
        return self.active_notebook.id

    @property
    def notebook_title(self) -> str:
        return self.active_notebook.title

    def switch_notebook(self, notebook: Notebook) -> None:
        """Switches the active Notebook aggregate, updating workspace and notes paths."""
        self.active_notebook = notebook
        self.active_notebook.ensure_directories()
        self.workspace_dir = self.active_notebook.workspace_dir
        self.watcher = FolderWatcher(self.workspace_dir)
        self.studio.set_notes_dir(self.active_notebook.notes_dir)
        self.desk.clear()
        self.sources.clear()
        self.initial_sync()

    @property
    def all_documents(self) -> Dict[str, ParsedDocument]:
        """Alias preserving backward compatibility with existing callers."""
        return self.sources

    def initial_sync(self):
        """Synchronizes Sources from the Notebook workspace directory using the SQLite FTS5 cache when unchanged."""
        for file_path in sorted(self.workspace_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in FolderWatcher.SUPPORTED_EXTENSIONS:
                if not file_path.name.startswith(".") and file_path.name not in self.sources:
                    try:
                        fhash = DocumentParser.calculate_file_hash(file_path)
                        cached_doc = self.archive.load_cached_document(file_path, fhash)
                        if cached_doc is not None and file_path.suffix.lower() not in DocumentParser.IMAGE_EXTENSIONS:
                            parsed = cached_doc
                        else:
                            parsed = DocumentParser.parse(file_path)
                            self.archive.add_document(parsed)
                        self.sources[parsed.source_file] = parsed
                        self.desk.add_document(parsed)
                    except Exception:
                        pass

    def _register_parsed_source(self, parsed: ParsedDocument) -> str:
        self.sources[parsed.source_file] = parsed
        self.desk.add_document(parsed)
        self.archive.add_document(parsed)
        return parsed.source_file

    @property
    def is_active_model_multimodal(self) -> bool:
        """Returns True if the currently active model preset supports multimodal vision/media."""
        return bool(self.engine.preset.supports_vision)

    def ingest_file_bytes(self, filename: str, content: bytes) -> str:
        """Writes raw bytes into the Notebook workspace, parses the Source, and indexes it into Desk and Archive."""
        safe_name = Path(filename).name
        ext = Path(safe_name).suffix.lower()
        if not self.is_active_model_multimodal and ext in (DocumentParser.IMAGE_EXTENSIONS | DocumentParser.AUDIO_EXTENSIONS):
            raise ValueError(
                f"El modelo activo ({self.engine.preset.name}) es de solo texto y no admite adjuntar archivos multimedia ({ext}). "
                f"Cambia a un modelo Multimodal (ej. Gemma 4 E2B o 12B) para subir imágenes o audio."
            )
        dest = (self.workspace_dir / safe_name).resolve()
        if dest.parent != self.workspace_dir:
            raise ValueError("Invalid destination filename")
        dest.write_bytes(content)
        parsed = DocumentParser.parse(dest)
        return self._register_parsed_source(parsed)

    def ingest_url(self, url: str) -> str:
        """Fetches a web URL or YouTube transcript and registers it as a Notebook Source."""
        dest = DocumentParser.ingest_url_to_file(url, self.workspace_dir)
        parsed = DocumentParser.parse(dest)
        return self._register_parsed_source(parsed)

    def ingest_pasted_text(self, title: str, content: str) -> str:
        """Creates a Markdown Source from pasted text and indexes it into Desk and Archive."""
        safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", title.strip())[:45].strip("_") or "Texto_Pegado"
        dest = self.workspace_dir / f"{safe_title}.md"
        dest.write_text(f"# {title.strip() or 'Texto Pegado'}\n\n{content}", encoding="utf-8")
        parsed = DocumentParser.parse(dest)
        return self._register_parsed_source(parsed)

    def get_sources_list(self) -> List[Dict]:
        return [
            {
                "name": name,
                "chunks_count": len(doc.chunks),
                "size_chars": len(doc.full_text),
                "media_type": doc.media_type,
            }
            for name, doc in sorted(self.sources.items())
        ]

    def get_source_detail(self, name: str) -> Dict:
        doc = self.sources.get(name)
        if not doc:
            return {"name": name, "chunks_count": 0, "preview": "Fuente no encontrada.", "media_type": "text"}
        return {
            "name": name,
            "chunks_count": len(doc.chunks),
            "preview": doc.full_text[:15000],
            "media_type": doc.media_type,
            "image_data_uri": doc.image_data_uri,
        }

    def sync_selected(self, selected: List[str]):
        """Updates which Sources reside in the active Desk (Tier 1) versus only in Archive (Tier 2)."""
        self.desk.clear()
        for name, doc in self.sources.items():
            if name in selected:
                self.desk.add_document(doc)
            else:
                self.archive.add_document(doc)

    def remove_source(self, filename: str) -> bool:
        """Removes a Source from Desk, Archive, and the workspace directory."""
        if filename in self.sources:
            del self.sources[filename]
        self.desk.remove_document(filename)
        self.archive.remove_document(filename)
        if filename in self.watcher.file_hashes:
            del self.watcher.file_hashes[filename]
            self.watcher._save_state()
        file_path = (self.workspace_dir / Path(filename).name).resolve()
        if file_path.parent == self.workspace_dir and file_path.exists():
            file_path.unlink()
            return True
        return False

    def clear_all_sources(self):
        for name in list(self.sources.keys()):
            self.remove_source(name)

    def reset_notebook(self, clear_notes: bool = True) -> Dict:
        """Resets the notebook workspace: clears active sources, desk budget, and clears notes for the new notebook."""
        self.clear_all_sources()
        if clear_notes:
            self.studio.clear_all_notes()
        return {
            "title": "Cuaderno sin título",
            "sources": self.get_sources_list(),
            "notes": self.get_notes_list(),
            "selected": self.desk.get_active_files(),
        }

    def get_collections_list(self) -> List[Dict]:
        return self.catalog.list_collections()

    def load_collection(self, collection_id: str, replace_existing: bool = False) -> List[str]:
        src_files = self.catalog.resolve_collection_files(collection_id)
        if replace_existing:
            self.clear_all_sources()

        imported = []
        for src_file in src_files:
            dest = self.workspace_dir / src_file.name
            shutil.copy2(src_file, dest)
            parsed = DocumentParser.parse(dest)
            imported.append(self._register_parsed_source(parsed))
        return imported

    def get_samples_list(self) -> List[Dict]:
        return self.catalog.list_samples(set(self.sources.keys()))

    def load_samples(self, filenames: List[str], replace_existing: bool = False) -> List[str]:
        src_files = self.catalog.resolve_sample_files(filenames)
        if replace_existing:
            self.clear_all_sources()

        imported = []
        for src in src_files:
            dest = self.workspace_dir / src.name
            shutil.copy2(src, dest)
            parsed = DocumentParser.parse(dest)
            imported.append(self._register_parsed_source(parsed))
        return imported

    def get_notes_list(self) -> List[Dict]:
        return self.studio.list_notes()

    def save_note(
        self,
        title: str,
        content: str,
        note_id: Optional[str] = None,
        original_title: Optional[str] = None
    ) -> List[Dict]:
        self.studio.save_note(title, content, note_id=note_id, original_title=original_title)
        return self.studio.list_notes()

    def generate_note_title(self, content: str) -> str:
        return self.engine.generate_note_title(content)

    def remove_note(self, note_id_or_title: str) -> bool:
        return self.studio.delete_note(note_id_or_title)

    def convert_note_to_source(self, note_id_or_title: str) -> str:
        """Converts an existing Studio Note into an active Source in the workspace."""
        note = self.studio.get_note(note_id_or_title)
        if not note:
            raise ValueError(f"Note not found: {note_id_or_title}")

        safe_title = re.sub(r"[^\w\-\.]+", "_", note["title"]).strip("_") or "Nota"
        filename = f"{safe_title}.md"
        dest = self.workspace_dir / filename

        # Ensure unique destination name if file already exists
        counter = 1
        while dest.exists():
            filename = f"{safe_title}_{counter}.md"
            dest = self.workspace_dir / filename
            counter += 1

        dest.write_text(note["content"], encoding="utf-8")
        parsed = DocumentParser.parse(dest)
        return self._register_parsed_source(parsed)

    def generate_notebook_overview(self) -> Dict[str, str]:
        """Generates a concise summary of active sources and an inferred notebook title."""
        active_sources = self.desk.get_active_files()
        if not active_sources:
            return {
                "title": "Cuaderno de Investigación",
                "overview": "No hay fuentes seleccionadas. Sube o selecciona documentos para comenzar el análisis."
            }

        desk_preview = self.desk.build_desk_prefix(max_chars=6000)
        return self.engine.generate_notebook_overview(desk_preview)

    def stream_synthesis(
        self,
        message: str,
        action_type: str = "chat",
        selected: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        image_uri: Optional[str] = None,
    ) -> Iterator[str]:
        """Executes a multimodal chat or Studio Artifact synthesis combining Desk prefix and Archive citations."""
        if selected is not None:
            self.sync_selected(selected)

        ctx_len = self.engine.preset.context_length
        base_system_context, current_turn_prompt, active_images = StudioSynthesizer.assemble_prompt(
            desk=self.desk,
            archive=self.archive,
            message=message,
            action_type=action_type,
            context_length=ctx_len,
            image_uri=image_uri,
        )

        return self.engine.generate_chat_stream(
            system_prompt=base_system_context,
            user_prompt=current_turn_prompt,
            history=history,
            image_uris=active_images,
        )

    def run_deep_research(self, topic: str) -> str:
        return self.researcher.compile_research_dossier(topic)

    def import_deep_research(self, topic: str, dossier: str) -> str:
        path = self.researcher.import_to_workspace(topic, dossier)
        parsed = DocumentParser.parse(path)
        return self._register_parsed_source(parsed)
