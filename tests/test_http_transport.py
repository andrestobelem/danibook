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


def test_http_multi_notebook_lifecycle(tmp_path: Path):
    client, _ = _make_test_client(tmp_path)

    # Initial list contains default
    res_list = client.get("/api/notebooks")
    assert res_list.status_code == 200
    assert res_list.json()["active_notebook_id"] == "default"
    assert len(res_list.json()["notebooks"]) == 1

    # Create a new isolated Notebook
    res_create = client.post("/api/notebooks/create", json={"title": "Proyecto Cervantes"})
    assert res_create.status_code == 200
    created_id = res_create.json()["notebook_id"]
    assert created_id != "default"
    assert res_create.json()["notebook_title"] == "Proyecto Cervantes"

    # Add a Source inside the new Notebook
    res_paste = client.post("/api/sources/paste", json={"title": "Quijote_Nota", "content": "En un lugar de la Mancha."})
    assert res_paste.status_code == 200
    assert len(res_paste.json()["sources"]) == 1

    # Switch back to default -> Sources are isolated
    res_switch_def = client.post("/api/notebooks/switch", json={"notebook_id": "default"})
    assert res_switch_def.status_code == 200
    assert res_switch_def.json()["notebook_id"] == "default"
    assert res_switch_def.json()["sources"] == []

    # Switch back to created Notebook -> Source is still there
    res_switch_nb = client.post("/api/notebooks/switch", json={"notebook_id": created_id})
    assert res_switch_nb.status_code == 200
    assert len(res_switch_nb.json()["sources"]) == 1

    # Rename active Notebook
    res_rename = client.post("/api/notebooks/rename", json={"title": "Cervantes II"})
    assert res_rename.status_code == 200
    assert res_rename.json()["notebook_title"] == "Cervantes II"

    # Switching to unknown Notebook returns 404
    res_404 = client.post("/api/notebooks/switch", json={"notebook_id": "does-not-exist"})
    assert res_404.status_code == 404
