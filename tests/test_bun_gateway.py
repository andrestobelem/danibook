import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import uvicorn

from danibook.domain import NotebookService
from danibook.ingestion.parser import DocumentChunk, ParsedDocument
from danibook.llm.engine import LLMEngine
from danibook.memory.tier2_archive import ArchiveMemory
from danibook.ui.app import create_fastapi_app


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _resolve_bun_bin() -> str | None:
    found = shutil.which("bun")
    if found:
        return found
    home_bun = Path.home() / ".bun" / "bin" / "bun"
    return str(home_bun) if home_bun.exists() else None


def test_bun_gateway_fts_and_proxy(tmp_path: Path):
    bun_bin = _resolve_bun_bin()
    if not bun_bin:
        pytest.skip("bun binary not installed")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = ArchiveMemory(db_path=cache_dir / "archive_fts5.sqlite")

    doc = ParsedDocument(
        source_file="arquitectura_bun.md",
        file_path=tmp_path / "workspace" / "arquitectura_bun.md",
        file_hash="hash-bun-test",
        chunks=[
            DocumentChunk(
                source_file="arquitectura_bun.md",
                content="La arquitectura híbrida de DaniBookLM combina Bun y SQLite FTS5.",
                page_number=1,
                section_title="Resumen",
            )
        ],
        full_text="La arquitectura híbrida de DaniBookLM combina Bun y SQLite FTS5.",
    )
    archive.add_document(doc)

    service = NotebookService(
        workspace_dir=tmp_path / "workspace",
        notes_dir=tmp_path / "notes",
        collections_dir=tmp_path / "collections",
        samples_dir=tmp_path / "samples",
        archive=archive,
        engine=LLMEngine(use_mock=True),
    )
    app = create_fastapi_app(notebook=service)
    py_port = _find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=py_port, log_level="warning")
    py_server = uvicorn.Server(config)
    py_thread = threading.Thread(target=py_server.run, daemon=True)
    py_thread.start()

    bun_port = _find_free_port()
    env = os.environ.copy()
    env["DANIBOOK_BUN_PORT"] = str(bun_port)
    env["DANIBOOK_PYTHON_URL"] = f"http://127.0.0.1:{py_port}"
    env["DANIBOOK_DATA_DIR"] = str(tmp_path)

    repo_root = Path(__file__).resolve().parent.parent
    bun_proc = subprocess.Popen(
        [bun_bin, "run", "src/bun/server.ts"],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    bun_base = f"http://127.0.0.1:{bun_port}"
    try:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{bun_base}/api/health", timeout=1) as resp:
                    if resp.status == 200:
                        health = json.loads(resp.read().decode("utf-8"))
                        break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("Bun gateway failed to start in time")

        assert health["status"] == "ok"
        assert health["runtime"] == "bun"
        assert health["fts_active"] is True

        # 1. Shared FTS5 search finds document indexed by Python ArchiveMemory
        with urllib.request.urlopen(f"{bun_base}/api/fts/search?q=Arquitectura", timeout=2) as resp:
            fts_data = json.loads(resp.read().decode("utf-8"))
        assert fts_data["count"] >= 1
        assert fts_data["results"][0]["source_file"] == "arquitectura_bun.md"
        assert fts_data["results"][0]["score"] > 0

        # 2. Reverse proxy forwards /api/state to Python backend
        with urllib.request.urlopen(f"{bun_base}/api/state", timeout=2) as resp:
            state_data = json.loads(resp.read().decode("utf-8"))
        assert state_data["notebook_id"] == service.notebook_id

        # 3. Path traversal on /vendor/ is blocked with 404
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{bun_base}/vendor/%2e%2e/server.ts", timeout=2)
        assert exc_info.value.code == 404
    finally:
        bun_proc.terminate()
        try:
            bun_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bun_proc.kill()
        py_server.should_exit = True
        py_thread.join(timeout=5)
