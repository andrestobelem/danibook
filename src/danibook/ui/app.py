from pathlib import Path
from typing import List, Dict, Optional
import re
import uvicorn
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import WORKSPACE_DIR, NOTES_DIR, LOCAL_GGUF_PATH, DEFAULT_MODEL_ID, COLLECTIONS_DIR, SAMPLES_DIR
from ..ingestion.parser import DocumentParser, ParsedDocument
from ..ingestion.watcher import FolderWatcher
from ..ingestion.researcher import DeepResearcher
from ..memory.tier1_desk import DeskMemory
from ..memory.tier2_archive import ArchiveMemory
from ..llm.engine import LLMEngine
from ..llm.prompts import SYSTEM_AUTHOR_PROMPT, build_chat_prompt


class SelectSourcesPayload(BaseModel):
    selected: List[str]


class ChatStreamPayload(BaseModel):
    message: str
    action_type: str = "chat"
    selected: Optional[List[str]] = None
    history: Optional[List[Dict[str, str]]] = None
    image_uri: Optional[str] = None


class SaveNotePayload(BaseModel):
    title: str
    content: str


class ResearchRunPayload(BaseModel):
    topic: str


class ResearchImportPayload(BaseModel):
    topic: str
    dossier: str


class AddUrlSourcePayload(BaseModel):
    url: str


class AddPastedTextPayload(BaseModel):
    title: str
    content: str


class SwitchModelPayload(BaseModel):
    model_id: str


class LoadCollectionPayload(BaseModel):
    collection_id: str
    replace_existing: bool = False


class LoadSamplesPayload(BaseModel):
    filenames: List[str]
    replace_existing: bool = False


class DanibookServer:
    def __init__(self):
        self.watcher = FolderWatcher(WORKSPACE_DIR)
        self.desk = DeskMemory()
        self.archive = ArchiveMemory()
        self.researcher = DeepResearcher(max_results=5)
        self.engine = LLMEngine(
            model_path=LOCAL_GGUF_PATH,
            use_mock=not LOCAL_GGUF_PATH.exists(),
            model_id=DEFAULT_MODEL_ID,
        )
        self.all_documents: Dict[str, ParsedDocument] = {}
        self.initial_sync()

    def initial_sync(self):
        for file_path in sorted(WORKSPACE_DIR.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in FolderWatcher.SUPPORTED_EXTENSIONS:
                if not file_path.name.startswith(".") and file_path.name not in self.all_documents:
                    try:
                        fhash = DocumentParser.calculate_file_hash(file_path)
                        cached_doc = self.archive.load_cached_document(file_path, fhash)
                        if cached_doc is not None and file_path.suffix.lower() not in DocumentParser.IMAGE_EXTENSIONS:
                            parsed = cached_doc
                        else:
                            parsed = DocumentParser.parse(file_path)
                            self.archive.add_document(parsed)
                        self.all_documents[parsed.source_file] = parsed
                        self.desk.add_document(parsed)
                    except Exception:
                        pass

    def get_sources_list(self) -> List[Dict]:
        return [
            {
                "name": name,
                "chunks_count": len(doc.chunks),
                "size_chars": len(doc.full_text),
                "media_type": doc.media_type,
            }
            for name, doc in sorted(self.all_documents.items())
        ]

    def get_notes_list(self) -> List[Dict]:
        notes = []
        for p in sorted(NOTES_DIR.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                notes.append({
                    "title": p.stem,
                    "content": p.read_text(encoding="utf-8"),
                })
            except Exception:
                pass
        return notes

    def sync_selected(self, selected: List[str]):
        self.desk.clear()
        for name, doc in self.all_documents.items():
            if name in selected:
                self.desk.add_document(doc)
            else:
                self.archive.add_document(doc)

    def remove_source(self, filename: str) -> bool:
        if filename in self.all_documents:
            del self.all_documents[filename]
        self.desk.remove_document(filename)
        self.archive.remove_document(filename)
        if filename in self.watcher.file_hashes:
            del self.watcher.file_hashes[filename]
            self.watcher._save_state()
        file_path = (WORKSPACE_DIR / Path(filename).name).resolve()
        if file_path.parent == WORKSPACE_DIR.resolve() and file_path.exists():
            file_path.unlink()
            return True
        return False

    def remove_note(self, title: str) -> bool:
        note_path = (NOTES_DIR / f"{Path(title).name}.md").resolve()
        if note_path.parent == NOTES_DIR.resolve() and note_path.exists():
            note_path.unlink()
            return True
        return False

    def get_collections_list(self) -> List[Dict]:
        collections_info = {
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
        result = []
        for col_dir in sorted(COLLECTIONS_DIR.iterdir()):
            if col_dir.is_dir() and not col_dir.name.startswith("."):
                cid = col_dir.name
                files = [f.name for f in col_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
                meta = collections_info.get(cid, {
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

    def load_collection(self, collection_id: str, replace_existing: bool = False) -> List[str]:
        target_dir = COLLECTIONS_DIR / collection_id
        if not target_dir.exists() or not target_dir.is_dir():
            raise ValueError(f"Collection not found: {collection_id}")

        import shutil
        if replace_existing:
            # Clear current workspace sources
            for name in list(self.all_documents.keys()):
                self.remove_source(name)

        imported = []
        for src_file in target_dir.iterdir():
            if src_file.is_file() and not src_file.name.startswith("."):
                dest = WORKSPACE_DIR / src_file.name
                shutil.copy2(src_file, dest)
                parsed = DocumentParser.parse(dest)
                self.all_documents[parsed.source_file] = parsed
                self.desk.add_document(parsed)
                self.archive.add_document(parsed)
                imported.append(parsed.source_file)
        return imported

    def get_samples_list(self) -> List[Dict]:
        samples = []
        if SAMPLES_DIR.exists():
            for p in sorted(SAMPLES_DIR.iterdir()):
                if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in FolderWatcher.SUPPORTED_EXTENSIONS:
                    size_kb = p.stat().st_size / 1024.0
                    size_str = f"{size_kb / 1024.0:.2f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
                    samples.append({
                        "filename": p.name,
                        "size": size_str,
                        "ext": p.suffix.lower(),
                        "is_in_workspace": p.name in self.all_documents,
                    })
        return samples

    def load_samples(self, filenames: List[str], replace_existing: bool = False) -> List[str]:
        import shutil
        if replace_existing:
            for name in list(self.all_documents.keys()):
                self.remove_source(name)

        imported = []
        for name in filenames:
            src = SAMPLES_DIR / name
            if src.exists() and src.is_file():
                dest = WORKSPACE_DIR / name
                shutil.copy2(src, dest)
                parsed = DocumentParser.parse(dest)
                self.all_documents[parsed.source_file] = parsed
                self.desk.add_document(parsed)
                self.archive.add_document(parsed)
                imported.append(parsed.source_file)
        return imported


def create_fastapi_app() -> FastAPI:
    server = DanibookServer()
    api = FastAPI(title="Danibook — NotebookLM Local")
    ui_dir = Path(__file__).parent
    html_path = ui_dir / "notebooklm.html"
    api.mount("/vendor", StaticFiles(directory=str(ui_dir / "vendor")), name="vendor")

    @api.get("/", response_class=HTMLResponse)
    async def index():
        return html_path.read_text(encoding="utf-8")

    @api.get("/api/state")
    async def get_state():
        return {
            "sources": server.get_sources_list(),
            "selected": server.desk.get_active_files(),
            "notes": server.get_notes_list(),
            "models": server.engine.get_models_status(),
            "active_model_id": server.engine.active_model_id,
            "vision_enabled": server.engine._vision_enabled,
            "collections": server.get_collections_list(),
            "samples": server.get_samples_list(),
        }

    @api.get("/api/samples")
    async def get_samples():
        return {"samples": server.get_samples_list()}

    @api.post("/api/samples/load")
    async def load_samples_endpoint(payload: LoadSamplesPayload):
        try:
            imported = server.load_samples(payload.filenames, replace_existing=payload.replace_existing)
            return {
                "status": "ok",
                "imported": imported,
                "sources": server.get_sources_list(),
                "selected": server.desk.get_active_files(),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.get("/api/collections")
    async def get_collections():
        return {"collections": server.get_collections_list()}

    @api.post("/api/collections/load")
    async def load_collection_endpoint(payload: LoadCollectionPayload):
        try:
            imported = server.load_collection(payload.collection_id, replace_existing=payload.replace_existing)
            return {
                "status": "ok",
                "imported": imported,
                "sources": server.get_sources_list(),
                "selected": server.desk.get_active_files(),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.get("/api/models")
    async def get_models():
        return {
            "models": server.engine.get_models_status(),
            "active_model_id": server.engine.active_model_id,
            "vision_enabled": server.engine._vision_enabled,
            "download_state": server.engine.download_state,
        }

    @api.get("/api/models/progress")
    async def get_models_progress():
        return server.engine.download_state

    @api.post("/api/models/select")
    def select_model(payload: SwitchModelPayload):
        try:
            info = server.engine.switch_model(payload.model_id, auto_download=True)
            return {
                "status": "ok",
                **info,
                "models": server.engine.get_models_status(),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.delete("/api/models/delete")
    async def delete_model(model_id: str = Query(...)):
        try:
            res = server.engine.delete_model_from_disk(model_id)
            return {"status": "ok", **res}
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.post("/api/models/unload")
    async def unload_model():
        server.engine.unload_active_model()
        return {
            "status": "ok",
            "models": server.engine.get_models_status(),
            "active_model_id": server.engine.active_model_id,
        }

    @api.post("/api/sources/select")
    async def select_sources(payload: SelectSourcesPayload):
        server.sync_selected(payload.selected)
        return {"status": "ok", "selected": server.desk.get_active_files()}

    @api.get("/api/sources/content")
    async def get_source_content(name: str = Query(...)):
        doc = server.all_documents.get(name)
        if not doc:
            return {"name": name, "chunks_count": 0, "preview": "Fuente no encontrada.", "media_type": "text"}
        return {
            "name": name,
            "chunks_count": len(doc.chunks),
            "preview": doc.full_text[:15000],
            "media_type": doc.media_type,
            "image_data_uri": doc.image_data_uri,
        }

    @api.post("/api/sources/upload")
    async def upload_sources(files: List[UploadFile] = File(...)):
        uploaded_names = []
        for f in files:
            dest = WORKSPACE_DIR / f.filename
            content = await f.read()
            dest.write_bytes(content)
            parsed = DocumentParser.parse(dest)
            server.all_documents[parsed.source_file] = parsed
            server.desk.add_document(parsed)
            server.archive.add_document(parsed)
            uploaded_names.append(parsed.source_file)
        return {
            "uploaded": uploaded_names,
            "sources": server.get_sources_list(),
        }

    @api.post("/api/sources/url")
    async def add_url_source(payload: AddUrlSourcePayload):
        try:
            dest = DocumentParser.ingest_url_to_file(payload.url, WORKSPACE_DIR)
            parsed = DocumentParser.parse(dest)
            server.all_documents[parsed.source_file] = parsed
            server.desk.add_document(parsed)
            server.archive.add_document(parsed)
            return {
                "status": "ok",
                "imported": parsed.source_file,
                "sources": server.get_sources_list(),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.post("/api/sources/paste")
    async def add_pasted_source(payload: AddPastedTextPayload):
        safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", payload.title.strip())[:45].strip("_") or "Texto_Pegado"
        dest = WORKSPACE_DIR / f"{safe_title}.md"
        dest.write_text(f"# {payload.title.strip() or 'Texto Pegado'}\n\n{payload.content}", encoding="utf-8")
        parsed = DocumentParser.parse(dest)
        server.all_documents[parsed.source_file] = parsed
        server.desk.add_document(parsed)
        server.archive.add_document(parsed)
        return {
            "status": "ok",
            "imported": parsed.source_file,
            "sources": server.get_sources_list(),
        }

    @api.delete("/api/sources/delete")
    async def delete_source(name: str = Query(...)):
        server.remove_source(name)
        return {
            "deleted": name,
            "sources": server.get_sources_list(),
            "selected": server.desk.get_active_files(),
        }

    @api.delete("/api/notes/delete")
    async def delete_note(title: str = Query(...)):
        server.remove_note(title)
        return {
            "deleted": title,
            "notes": server.get_notes_list(),
        }

    @api.post("/api/chat/stream")
    async def chat_stream(payload: ChatStreamPayload):
        if payload.selected is not None:
            server.sync_selected(payload.selected)

        desk_prefix = server.desk.build_desk_prefix(max_chars=4500)
        active_images = server.desk.get_active_image_uris(max_images=2)
        if payload.image_uri and payload.image_uri not in active_images:
            active_images.insert(0, payload.image_uri)
        reference_context = server.archive.format_search_context(payload.message, top_k=3)

        base_system_context = SYSTEM_AUTHOR_PROMPT
        if desk_prefix:
            base_system_context += f"\n\n{desk_prefix}"

        current_turn_prompt = build_chat_prompt(
            desk_prefix="",
            reference_context=reference_context,
            user_query=payload.message,
            action_type=payload.action_type,
        )

        def token_generator():
            for token in server.engine.generate_chat_stream(
                system_prompt=base_system_context,
                user_prompt=current_turn_prompt,
                history=payload.history,
                image_uris=active_images,
            ):
                yield token

        return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")

    @api.post("/api/notes/save")
    async def save_note(payload: SaveNotePayload):
        clean_title = "".join(c for c in payload.title if c.isalnum() or c in ("-", "_", " ")).strip().replace(" ", "_")
        if not clean_title:
            clean_title = "nota_cuaderno"
        note_path = NOTES_DIR / f"{clean_title}.md"
        note_path.write_text(payload.content, encoding="utf-8")
        return {"status": "ok", "notes": server.get_notes_list()}

    @api.post("/api/research/run")
    async def run_research(payload: ResearchRunPayload):
        dossier = server.researcher.compile_research_dossier(payload.topic)
        return {"dossier": dossier}

    @api.post("/api/research/import")
    async def import_research(payload: ResearchImportPayload):
        path = server.researcher.import_to_workspace(payload.topic, payload.dossier)
        parsed = DocumentParser.parse(path)
        server.all_documents[parsed.source_file] = parsed
        server.desk.add_document(parsed)
        server.archive.add_document(parsed)
        return {
            "imported": parsed.source_file,
            "sources": server.get_sources_list(),
        }

    return api


def main():
    api = create_fastapi_app()
    uvicorn.run(api, host="127.0.0.1", port=7860, log_level="info")


if __name__ == "__main__":
    main()
