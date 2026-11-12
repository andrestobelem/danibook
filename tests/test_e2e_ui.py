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
def live_app_url(tmp_path: Path):
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
        yield f"{base_url}/app"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_e2e_sources_chat_studio_notes_and_models(page: Page, live_app_url: str):
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
