from pathlib import Path
from fastapi.testclient import TestClient

from danibook.domain import NotebookService
from danibook.llm.engine import LLMEngine
from danibook.memory.tier2_archive import ArchiveMemory
from danibook.ui.app import create_fastapi_app


def _make_test_client(tmp_path: Path) -> tuple[TestClient, NotebookService]:
    service = NotebookService(
        workspace_dir=tmp_path / "workspace",
        notes_dir=tmp_path / "notes",
        collections_dir=tmp_path / "collections",
        samples_dir=tmp_path / "samples",
        archive=ArchiveMemory(db_path=tmp_path / "archive.sqlite"),
        engine=LLMEngine(use_mock=True),
    )
    app = create_fastapi_app(notebook=service)
    return TestClient(app, raise_server_exceptions=False), service


def test_http_state_and_models_use_domain_facade(tmp_path: Path):
    client, service = _make_test_client(tmp_path)

    res_state = client.get("/api/state")
    assert res_state.status_code == 200
    data = res_state.json()
    assert data["notebook_id"] == service.notebook_id
    assert data["vision_enabled"] == service.is_active_model_multimodal
    assert data["selected"] == service.get_desk_sources()

    res_models = client.get("/api/models")
    assert res_models.status_code == 200
    models_data = res_models.json()
    assert models_data["vision_enabled"] == service.is_active_model_multimodal
    assert models_data["active_model_id"] == service.engine.active_model_id


def test_http_error_handlers_map_validation_and_unhandled_errors(tmp_path: Path, monkeypatch):
    client, service = _make_test_client(tmp_path)

    # ValueError -> 400
    res_400 = client.post("/api/notes/convert-to-source", json={})
    assert res_400.status_code == 400
    assert res_400.json()["status"] == "error"

    res_del_400 = client.delete("/api/models/delete", params={"model_id": "nonexistent-model"})
    assert res_del_400.status_code == 400
    assert res_del_400.json()["status"] == "error"

    # Unhandled Exception -> 500 JSON
    def boom(_topic: str):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(service, "run_deep_research", boom)
    res_500 = client.post("/api/research/run", json={"topic": "test"})
    assert res_500.status_code == 500
    assert res_500.json() == {"status": "error", "message": "unexpected failure"}


def test_http_paste_and_delete_source_roundtrip(tmp_path: Path):
    client, _ = _make_test_client(tmp_path)

    res_paste = client.post(
        "/api/sources/paste",
        json={"title": "Mi Fuente", "content": "Contenido de prueba."},
    )
    assert res_paste.status_code == 200
    imported = res_paste.json()["imported"]
    assert imported in [s["name"] for s in res_paste.json()["sources"]]

    res_del = client.delete("/api/sources/delete", params={"name": imported})
    assert res_del.status_code == 200
    assert res_del.json()["deleted"] == imported
    assert imported not in res_del.json()["selected"]
