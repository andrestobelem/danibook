from pathlib import Path
from typing import List, Dict, Optional
import uvicorn
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..domain.notebook import NotebookService


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
    api.mount("/vendor", StaticFiles(directory=str(ui_dir / "vendor")), name="vendor")

    @api.get("/", response_class=HTMLResponse)
    async def index():
        return html_path.read_text(encoding="utf-8")

    @api.get("/api/state")
    async def get_state():
        return {
            "sources": service.get_sources_list(),
            "selected": service.desk.get_active_files(),
            "notes": service.get_notes_list(),
            "models": service.engine.get_models_status(),
            "active_model_id": service.engine.active_model_id,
            "vision_enabled": service.engine._vision_enabled,
            "collections": service.get_collections_list(),
            "samples": service.get_samples_list(),
        }

    @api.get("/api/samples")
    async def get_samples():
        return {"samples": service.get_samples_list()}

    @api.post("/api/samples/load")
    async def load_samples_endpoint(payload: LoadSamplesPayload):
        try:
            imported = service.load_samples(payload.filenames, replace_existing=payload.replace_existing)
            return {
                "status": "ok",
                "imported": imported,
                "sources": service.get_sources_list(),
                "selected": service.desk.get_active_files(),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.get("/api/collections")
    async def get_collections():
        return {"collections": service.get_collections_list()}

    @api.post("/api/collections/load")
    async def load_collection_endpoint(payload: LoadCollectionPayload):
        try:
            imported = service.load_collection(payload.collection_id, replace_existing=payload.replace_existing)
            return {
                "status": "ok",
                "imported": imported,
                "sources": service.get_sources_list(),
                "selected": service.desk.get_active_files(),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.get("/api/models")
    async def get_models():
        return {
            "models": service.engine.get_models_status(),
            "active_model_id": service.engine.active_model_id,
            "vision_enabled": service.engine._vision_enabled,
            "download_state": service.engine.download_state,
        }

    @api.get("/api/models/progress")
    async def get_models_progress():
        return service.engine.download_state

    @api.post("/api/models/select")
    def select_model(payload: SwitchModelPayload):
        try:
            info = service.engine.switch_model(payload.model_id, auto_download=True)
            return {
                "status": "ok",
                **info,
                "models": service.engine.get_models_status(),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.delete("/api/models/delete")
    async def delete_model(model_id: str = Query(...)):
        try:
            res = service.engine.delete_model_from_disk(model_id)
            return {"status": "ok", **res}
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.post("/api/models/unload")
    async def unload_model():
        service.engine.unload_active_model()
        return {
            "status": "ok",
            "models": service.engine.get_models_status(),
            "active_model_id": service.engine.active_model_id,
        }

    @api.post("/api/sources/select")
    async def select_sources(payload: SelectSourcesPayload):
        service.sync_selected(payload.selected)
        return {"status": "ok", "selected": service.desk.get_active_files()}

    @api.get("/api/sources/content")
    async def get_source_content(name: str = Query(...)):
        return service.get_source_detail(name)

    @api.post("/api/sources/upload")
    async def upload_sources(files: List[UploadFile] = File(...)):
        try:
            uploaded_names = []
            for f in files:
                content = await f.read()
                name = service.ingest_file_bytes(f.filename, content)
                uploaded_names.append(name)
            return {
                "status": "ok",
                "uploaded": uploaded_names,
                "sources": service.get_sources_list(),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.post("/api/sources/url")
    async def add_url_source(payload: AddUrlSourcePayload):
        try:
            imported = service.ingest_url(payload.url)
            return {
                "status": "ok",
                "imported": imported,
                "sources": service.get_sources_list(),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.post("/api/notebook/new")
    async def new_notebook():
        data = service.reset_notebook()
        return {"status": "ok", **data}

    @api.post("/api/sources/paste")
    async def add_pasted_source(payload: AddPastedTextPayload):
        imported = service.ingest_pasted_text(payload.title, payload.content)
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
            "selected": service.desk.get_active_files(),
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
            yield from service.stream_synthesis(
                message=payload.message,
                action_type=payload.action_type,
                selected=payload.selected,
                history=payload.history,
                image_uri=payload.image_uri,
            )

        return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")

    @api.post("/api/notes/save")
    async def save_note(payload: SaveNotePayload):
        notes = service.save_note(payload.title, payload.content, note_id=payload.id, original_title=payload.original_title)
        return {"status": "ok", "notes": notes}

    @api.post("/api/notes/generate-title")
    async def generate_note_title_endpoint(payload: GenerateNoteTitlePayload):
        title = service.generate_note_title(payload.content)
        return {"status": "ok", "title": title}

    @api.post("/api/notes/convert-to-source")
    async def convert_note_to_source(payload: ConvertNoteToSourcePayload):
        target = payload.id or payload.title
        if not target:
            return JSONResponse(status_code=400, content={"status": "error", "message": "id or title is required"})
        try:
            source_name = service.convert_note_to_source(target)
            return {
                "status": "ok",
                "source_name": source_name,
                "sources": service.get_sources_list(),
                "selected": service.desk.get_active_files(),
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

    @api.post("/api/notebook/overview")
    async def generate_notebook_overview_endpoint():
        res = service.generate_notebook_overview()
        return {"status": "ok", **res}

    @api.post("/api/research/run")
    async def run_research(payload: ResearchRunPayload):
        dossier = service.run_deep_research(payload.topic)
        return {"dossier": dossier}

    @api.post("/api/research/import")
    async def import_research(payload: ResearchImportPayload):
        imported = service.import_deep_research(payload.topic, payload.dossier)
        return {
            "imported": imported,
            "sources": service.get_sources_list(),
        }

    return api


def main():
    api = create_fastapi_app()
    uvicorn.run(api, host="127.0.0.1", port=7860, log_level="info")


if __name__ == "__main__":
    main()
