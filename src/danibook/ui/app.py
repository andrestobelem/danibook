from pathlib import Path
from typing import List, Dict, Optional
import logging
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..domain.notebook import NotebookService

logger = logging.getLogger(__name__)


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
    id: Optional[str] = None
    original_title: Optional[str] = None


class ConvertNoteToSourcePayload(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None


class GenerateNoteTitlePayload(BaseModel):
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


# Backward-compatible alias for NotebookService
DanibookServer = NotebookService


def create_fastapi_app(notebook: Optional[NotebookService] = None) -> FastAPI:
    """Builds the HTTP transport layer delegating all domain rules to NotebookService."""
    service = notebook or NotebookService()
    api = FastAPI(title="DaniBookLM — Local Gemma 4 Multimodal")
    ui_dir = Path(__file__).parent
    html_path = ui_dir / "notebooklm.html"
    status_html_path = ui_dir / "daemon_status.html"
    api.mount("/vendor", StaticFiles(directory=str(ui_dir / "vendor")), name="vendor")

    @api.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("Validation error on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})

    @api.exception_handler(FileNotFoundError)
    async def not_found_handler(request: Request, exc: FileNotFoundError):
        logger.warning("Resource not found on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=404, content={"status": "error", "message": str(exc)})

    @api.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})

    @api.get("/", response_class=HTMLResponse)
    async def index():
        return status_html_path.read_text(encoding="utf-8")

    @api.get("/app", response_class=HTMLResponse)
    async def direct_app():
        return html_path.read_text(encoding="utf-8")

    @api.get("/api/state")
    async def get_state():
        return service.get_notebook_state()

    @api.get("/api/samples")
    async def get_samples():
        return {"samples": service.get_samples_list()}

    @api.post("/api/samples/load")
    async def load_samples_endpoint(payload: LoadSamplesPayload):
        imported = await run_in_threadpool(
            service.load_samples, payload.filenames, replace_existing=payload.replace_existing
        )
        return {
            "status": "ok",
            "imported": imported,
            "sources": service.get_sources_list(),
            "selected": service.get_desk_sources(),
        }

    @api.get("/api/collections")
    async def get_collections():
        return {"collections": service.get_collections_list()}

    @api.post("/api/collections/load")
    async def load_collection_endpoint(payload: LoadCollectionPayload):
        imported = await run_in_threadpool(
            service.load_collection, payload.collection_id, replace_existing=payload.replace_existing
        )
        return {
            "status": "ok",
            "imported": imported,
            "sources": service.get_sources_list(),
            "selected": service.get_desk_sources(),
        }

    @api.get("/api/models")
    async def get_models():
        return service.get_models_state()

    @api.get("/api/models/progress")
    async def get_models_progress():
        return service.get_model_download_progress()

    @api.post("/api/models/select")
    async def select_model(payload: SwitchModelPayload):
        info = await run_in_threadpool(service.select_model, payload.model_id, auto_download=True)
        return {"status": "ok", **info}

    @api.delete("/api/models/delete")
    async def delete_model(model_id: str = Query(...)):
        res = service.delete_model(model_id)
        return {"status": "ok", **res}

    @api.post("/api/models/unload")
    async def unload_model():
        res = service.unload_model()
        return {"status": "ok", **res}

    @api.post("/api/sources/select")
    async def select_sources(payload: SelectSourcesPayload):
        service.sync_selected(payload.selected)
        return {"status": "ok", "selected": service.get_desk_sources()}

    @api.get("/api/sources/content")
    async def get_source_content(name: str = Query(...)):
        return service.get_source_detail(name)

    @api.get("/api/fts/search")
    async def search_fts(q: str = Query("")):
        return service.search_archive(q)

    @api.post("/api/sources/upload")
    async def upload_sources(files: List[UploadFile] = File(...)):
        uploaded_names = []
        for f in files:
            content = await f.read()
            name = await run_in_threadpool(service.ingest_file_bytes, f.filename, content)
            uploaded_names.append(name)
        return {
            "status": "ok",
            "uploaded": uploaded_names,
            "sources": service.get_sources_list(),
        }

    @api.post("/api/sources/url")
    async def add_url_source(payload: AddUrlSourcePayload):
        imported = await run_in_threadpool(service.ingest_url, payload.url)
        return {
            "status": "ok",
            "imported": imported,
            "sources": service.get_sources_list(),
        }

    @api.post("/api/notebook/new")
    async def new_notebook():
        data = service.reset_notebook()
        return {"status": "ok", **data}

    @api.post("/api/sources/paste")
    async def add_pasted_source(payload: AddPastedTextPayload):
        imported = await run_in_threadpool(service.ingest_pasted_text, payload.title, payload.content)
        return {
            "status": "ok",
            "imported": imported,
            "sources": service.get_sources_list(),
        }

    @api.delete("/api/sources/delete")
    async def delete_source(name: str = Query(...)):
        service.remove_source(name)
        return {
            "deleted": name,
            "sources": service.get_sources_list(),
            "selected": service.get_desk_sources(),
        }

    @api.delete("/api/notes/delete")
    async def delete_note(id: Optional[str] = None, title: Optional[str] = None):
        target = id or title or ""
        service.remove_note(target)
        return {
            "deleted": target,
            "notes": service.get_notes_list(),
        }

    @api.post("/api/chat/stream")
    async def chat_stream(payload: ChatStreamPayload):
        def token_generator():
            try:
                yield from service.stream_synthesis(
                    message=payload.message,
                    action_type=payload.action_type,
                    selected=payload.selected,
                    history=payload.history,
                    image_uri=payload.image_uri,
                )
            except Exception as exc:
                logger.exception("Error during chat_stream synthesis")
                yield f"\n\n⚠️ Error de síntesis: {exc}"

        return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")

    @api.post("/api/notes/save")
    async def save_note(payload: SaveNotePayload):
        notes = service.save_note(payload.title, payload.content, note_id=payload.id, original_title=payload.original_title)
        return {"status": "ok", "notes": notes}

    @api.post("/api/notes/generate-title")
    async def generate_note_title_endpoint(payload: GenerateNoteTitlePayload):
        title = await run_in_threadpool(service.generate_note_title, payload.content)
        return {"status": "ok", "title": title}

    @api.post("/api/notes/convert-to-source")
    async def convert_note_to_source(payload: ConvertNoteToSourcePayload):
        target = payload.id or payload.title
        if not target:
            raise ValueError("id or title is required")
        source_name = await run_in_threadpool(service.convert_note_to_source, target)
        return {
            "status": "ok",
            "source_name": source_name,
            "sources": service.get_sources_list(),
            "selected": service.get_desk_sources(),
        }

    @api.post("/api/notebook/overview")
    async def generate_notebook_overview_endpoint():
        res = await run_in_threadpool(service.generate_notebook_overview)
        return {"status": "ok", **res}

    @api.post("/api/research/run")
    async def run_research(payload: ResearchRunPayload):
        dossier = await run_in_threadpool(service.run_deep_research, payload.topic)
        return {"dossier": dossier}

    @api.post("/api/research/import")
    async def import_research(payload: ResearchImportPayload):
        imported = await run_in_threadpool(service.import_deep_research, payload.topic, payload.dossier)
        return {
            "imported": imported,
            "sources": service.get_sources_list(),
        }

    return api


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    api = create_fastapi_app()
    uvicorn.run(api, host="127.0.0.1", port=7860, log_level="info")


if __name__ == "__main__":
    main()
