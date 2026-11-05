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
        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DaniBookLM — Local Inference Daemon (:7860)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: light dark;
      --bg: #131314;
      --card: #1e1f20;
      --border: #37393b;
      --text: #e3e3e3;
      --text-muted: #8e918f;
      --accent: #a8c7fa;
      --on-accent: #062e6f;
      --success: #34a853;
    }
    @media (prefers-color-scheme: light) {
      :root {
        --bg: #f8fafd;
        --card: #ffffff;
        --border: #e0e3e7;
        --text: #1f1f1f;
        --text-muted: #747775;
        --accent: #0b57d0;
        --on-accent: #ffffff;
      }
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', sans-serif;
      min-height: 100dvh;
      display: grid;
      place-content: center;
      padding: 24px;
    }
    .status-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 36px 32px;
      max-width: 540px;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      background: color-mix(in srgb, var(--success) 16%, transparent);
      color: var(--success);
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
      width: fit-content;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      animation: pulse 1.5s infinite alternate;
    }
    @keyframes pulse { from { opacity: 0.4; } to { opacity: 1; } }
    h1 {
      font-size: 1.45rem;
      font-weight: 700;
      line-height: 1.3;
    }
    p {
      color: var(--text-muted);
      font-size: 0.92rem;
      line-height: 1.6;
    }
    .code-pill {
      font-family: 'JetBrains Mono', monospace;
      background: color-mix(in srgb, var(--text-muted) 15%, transparent);
      padding: 2px 7px;
      border-radius: 6px;
      font-size: 0.85em;
      color: var(--text);
    }
    .btn-gateway {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      background: var(--accent);
      color: var(--on-accent);
      padding: 13px 22px;
      border-radius: 999px;
      text-decoration: none;
      font-size: 0.92rem;
      font-weight: 600;
      margin-top: 6px;
      transition: transform 0.15s ease, opacity 0.15s ease;
    }
    .btn-gateway:hover {
      transform: translateY(-2px);
      opacity: 0.94;
    }
    .meta-box {
      border-top: 1px dashed var(--border);
      padding-top: 14px;
      font-size: 0.8rem;
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
  </style>
</head>
<body>
  <main class="status-card">
    <div class="badge">
      <span class="badge-dot"></span>
      Metal Inference Socket Active
    </div>
    <h1>DaniBookLM — Inference Daemon (:7860)</h1>
    <p>
      This port operates as the <strong>internal backend socket</strong> for tensor computation with <span class="code-pill">llama-cpp-python</span> and Apple Metal GPU.
    </p>
    <p>
      To interact with the <strong>Research Notebook</strong> (user interface, FTS5 search, and multimodal chat), access through the primary gateway:
    </p>
    <a href="http://127.0.0.1:7861" class="btn-gateway">
      Open DaniBookLM (:7861) →
    </a>
    <div class="meta-box">
      <div>• Internal socket: <span class="code-pill">http://127.0.0.1:7860/api/*</span></div>
      <div>• Primary gateway: <span class="code-pill">http://127.0.0.1:7861</span></div>
      <div>• Direct UI fallback: <a href="/app" style="color:var(--accent);">/app</a></div>
    </div>
  </main>
</body>
</html>"""

    @api.get("/app", response_class=HTMLResponse)
    async def direct_app():
        return html_path.read_text(encoding="utf-8")

    @api.get("/api/state")
    async def get_state():
        return {
            "notebook_id": service.notebook_id,
            "notebook_title": service.notebook_title,
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
    async def select_model(payload: SwitchModelPayload):
        try:
            from starlette.concurrency import run_in_threadpool
            info = await run_in_threadpool(service.engine.switch_model, payload.model_id, auto_download=True)
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
