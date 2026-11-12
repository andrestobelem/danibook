import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import Page, expect

from danibook.domain import NotebookService
from danibook.llm.engine import LLMEngine
from danibook.memory.tier2_archive import ArchiveMemory
from danibook.ui.app import create_fastapi_app


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture()
def live_server(tmp_path: Path):
    """Starts a real Uvicorn server with an isolated NotebookService in mock LLM mode."""
    service = NotebookService(
        workspace_dir=tmp_path / "workspace",
        notes_dir=tmp_path / "notes",
        collections_dir=tmp_path / "collections",
        samples_dir=tmp_path / "samples",
        archive=ArchiveMemory(db_path=tmp_path / "archive.sqlite"),
        engine=LLMEngine(use_mock=True),
    )
    app = create_fastapi_app(notebook=service)
    port = _find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/state", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("Live test server failed to start in time")

    try:
        yield f"{base_url}/app", service
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_e2e_sources_chat_studio_notes_and_models(page: Page, live_server):
    live_app_url, _ = live_server
    page.goto(live_app_url, wait_until="domcontentloaded")

    # 1. Model selector is populated from /api/state
    model_select = page.locator("#model-select")
    expect(model_select).to_be_visible()
    expect(model_select.locator("option").first).to_be_attached()
    assert model_select.locator("option").count() >= 1

    # 2. Add Source via paste modal & verify Desk selection
    page.locator("#btn-open-add-source").click()
    expect(page.locator("#add-source-modal")).to_be_visible()
    page.locator("#paste-source-title").fill("E2E_Source")
    page.locator("#paste-source-content").fill("Contenido de prueba para el Notebook.")
    page.locator("#save-paste-source-btn").click()

    sources_list = page.locator("#sources-list")
    expect(sources_list).to_contain_text("E2E_Source.md")
    expect(page.locator("#active-sources-pill")).to_contain_text("1")

    # 2b. Archive FTS5 fragment search in Sources panel
    fts_input = page.locator("#archive-fts-input")
    fts_input.fill("Contenido")
    fts_input.press("Enter")
    fts_results = page.locator("#archive-fts-results")
    expect(fts_results).to_be_visible()
    expect(fts_results).to_contain_text("E2E_Source.md")
    fts_results.locator(".archive-fts-item").first.click()
    expect(page.locator("#source-modal")).to_be_visible()
    page.locator("#close-source-modal").click()

    # Toggle source checkbox off and back on
    source_checkbox = sources_list.locator('input[type="checkbox"]').first
    source_checkbox.uncheck()
    expect(page.locator("#active-sources-pill")).to_contain_text("0")
    source_checkbox.check()
    expect(page.locator("#active-sources-pill")).to_contain_text("1")

    # 3. Streaming chat
    page.locator("#chat-input").fill("¿Qué dice la fuente?")
    page.locator("#chat-send-btn").click()
    chat_stream = page.locator("#chat-stream")
    expect(chat_stream).to_contain_text("I analyzed the documents in your active desk.")

    # 4. Studio Artifact generation (Study Guide)
    page.locator('.studio-tile[data-studio="summary"]').click()
    expect(chat_stream).to_contain_text("Investigacion.epub")

    # 5. Studio Note creation & persistence across reload
    page.locator("#btn-new-note").click()
    page.locator("#note-title-input").fill("Nota E2E")
    page.locator("#note-content-input").fill("Resumen guardado desde Playwright.")
    page.locator("#save-note-confirm").click()

    notes_list = page.locator("#notes-list")
    expect(notes_list).to_contain_text("Nota E2E")

    page.reload(wait_until="domcontentloaded")
    expect(page.locator("#notes-list")).to_contain_text("Nota E2E")
    expect(page.locator("#sources-list")).to_contain_text("E2E_Source.md")


def test_e2e_mindmap_flashcards_and_markdown_export(page: Page, live_server, monkeypatch):
    live_app_url, service = live_server

    def fake_stream_synthesis(message: str, action_type: str = "chat", **_kwargs):
        if action_type == "mindmap":
            yield "```mermaid\ngraph TD\n  A[Source] --> B[Concept]\n```\n"
        elif action_type == "flashcards":
            yield (
                "### Card 1\n"
                "- **Front**: ¿Qué es Desk?\n"
                "- **Back**: Memoria Tier 1 en el prefijo del prompt.\n"
                "- **Takeaway**: Mantiene fuentes activas.\n"
            )
        else:
            yield "Respuesta base."

    monkeypatch.setattr(service, "stream_synthesis", fake_stream_synthesis)
    page.goto(live_app_url, wait_until="domcontentloaded")

    # 1. Visual Mind Map renders Mermaid SVG
    page.locator('.studio-tile[data-studio="mindmap"]').click()
    mermaid_svg = page.locator("#chat-stream .mermaid-container svg")
    expect(mermaid_svg).to_be_visible()

    # 2. Flashcards & Quiz renders interactive card and toggles reveal
    page.locator('.studio-tile[data-studio="flashcards"]').click()
    card = page.locator("#chat-stream .flashcard-item").first
    expect(card).to_be_visible()
    expect(card).to_contain_text("Click to reveal answer")
    card.click()
    expect(card).to_contain_text("Hide answer")
    expect(card).to_have_class(r"flashcard-item revealed")

    # 3. Notebook Export (Markdown download)
    page.locator("#btn-export-notebook").click()
    expect(page.locator("#export-notebook-modal")).to_be_visible()
    with page.expect_download() as download_info:
        page.locator("#btn-export-download-md").click()
    download = download_info.value
    assert download.suggested_filename.endswith(".md")

