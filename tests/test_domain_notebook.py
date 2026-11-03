from pathlib import Path
import pytest
from danibook.domain import NotebookService, StudioStore, CatalogRepository
from danibook.llm.engine import LLMEngine
from danibook.memory.tier2_archive import ArchiveMemory


def test_studio_store_lifecycle(tmp_path: Path):
    store = StudioStore(notes_dir=tmp_path / "notes")
    assert store.list_notes() == []

    saved_path = store.save_note("Capítulo 1: Introducción!", "Contenido inicial.")
    assert saved_path.exists()

    notes = store.list_notes()
    assert len(notes) == 1
    assert notes[0]["title"] == "Capítulo 1: Introducción!"
    assert notes[0]["content"] == "Contenido inicial."
    note_id = notes[0]["id"]
    assert note_id != ""

    # Renaming title keeps the same note id
    store.save_note("Capítulo 1: Revisado", "Contenido actualizado.", note_id=note_id)
    notes_updated = store.list_notes()
    assert len(notes_updated) == 1
    assert notes_updated[0]["id"] == note_id
    assert notes_updated[0]["title"] == "Capítulo 1: Revisado"
    assert notes_updated[0]["content"] == "Contenido actualizado."

    deleted = store.delete_note(note_id)
    assert deleted is True
    assert store.list_notes() == []


def test_catalog_repository_discovery(tmp_path: Path):
    cols_dir = tmp_path / "collections"
    col_a = cols_dir / "cervantes"
    col_a.mkdir(parents=True)
    (col_a / "quijote.md").write_text("# Quijote", encoding="utf-8")

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir(parents=True)
    (samples_dir / "muestra.md").write_text("# Muestra", encoding="utf-8")

    catalog = CatalogRepository(collections_dir=cols_dir, samples_dir=samples_dir)
    collections = catalog.list_collections()
    assert len(collections) == 1
    assert collections[0]["id"] == "cervantes"
    assert collections[0]["files"] == ["quijote.md"]

    resolved_col = catalog.resolve_collection_files("cervantes")
    assert len(resolved_col) == 1

    samples = catalog.list_samples(active_source_names={"muestra.md"})
    assert len(samples) == 1
    assert samples[0]["is_in_workspace"] is True


def test_notebook_service_orchestration(tmp_path: Path):
    ws = tmp_path / "workspace"
    notes = tmp_path / "notes"
    cols = tmp_path / "collections"
    samples = tmp_path / "samples"
    (cols / "ciencia_ia").mkdir(parents=True)
    (cols / "ciencia_ia" / "transformers.md").write_text("# Attention Is All You Need\nSelf-attention mechanism.", encoding="utf-8")

    service = NotebookService(
        workspace_dir=ws,
        notes_dir=notes,
        collections_dir=cols,
        samples_dir=samples,
        engine=LLMEngine(use_mock=True),
        archive=ArchiveMemory(db_path=tmp_path / "archive.sqlite"),
    )

    # 1. Ingest pasted Source
    source_name = service.ingest_pasted_text("Borrador Uno", "En 1999 comenzó el proyecto.")
    assert source_name == "Borrador_Uno.md"
    assert len(service.get_sources_list()) == 1

    # 2. Load Collection replacing existing
    imported = service.load_collection("ciencia_ia", replace_existing=True)
    assert imported == ["transformers.md"]
    assert [s["name"] for s in service.get_sources_list()] == ["transformers.md"]

    # 3. Stream synthesis
    tokens = list(service.stream_synthesis("Explica self-attention", action_type="summary"))
    assert len(tokens) > 0

    # 4. Studio Notes via NotebookService
    notes_list = service.save_note("Síntesis IA", "Resumen guardado.")
    assert len(notes_list) == 1
    assert notes_list[0]["title"] == "Síntesis IA"
    created_id = notes_list[0]["id"]

    # Convert note to source in workspace
    source_name = service.convert_note_to_source(created_id)
    assert source_name == "S_ntesis_IA.md" or "IA" in source_name
    assert any(s["name"] == source_name for s in service.get_sources_list())

    assert service.remove_note(created_id) is True


def test_multimodal_file_attachment_gating(tmp_path: Path):
    ws = tmp_path / "workspace"
    # Non-multimodal model (gemma-4-e4b supports_vision=False)
    text_only_service = NotebookService(
        workspace_dir=ws,
        notes_dir=tmp_path / "notes",
        collections_dir=tmp_path / "cols",
        samples_dir=tmp_path / "samples",
        engine=LLMEngine(use_mock=True, model_id="gemma-4-e4b"),
        archive=ArchiveMemory(db_path=tmp_path / "archive.sqlite"),
    )
    assert text_only_service.is_active_model_multimodal is False

    # Text files are allowed on text-only models
    name = text_only_service.ingest_file_bytes("apuntes.md", b"# Solo texto")
    assert name == "apuntes.md"

    # Image/Audio files are rejected on text-only models
    with pytest.raises(ValueError, match="solo texto"):
        text_only_service.ingest_file_bytes("foto.png", b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="solo texto"):
        text_only_service.ingest_file_bytes("audio.mp3", b"ID3")


def test_never_exceeds_context_window():
    # 1. Test clamping on 8K model (gemma-2-2b) and 32K model (gemma-4-e2b)
    for model_id in ("gemma-2-2b", "gemma-4-e2b"):
        engine = LLMEngine(use_mock=True, model_id=model_id)
        n_ctx = engine.preset.context_length
        huge_text = "Psicoanálisis de Freud y Lacan sobre el inconsciente estructurado. " * 5000  # ~330,000 chars

        max_budget = n_ctx // 2
        clamped = engine.clamp_to_token_budget(huge_text, max_budget)
        assert engine.count_tokens_safe(clamped) <= max_budget + 25  # Includes notice suffix

        # 2. Test history compaction with huge messages
        huge_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "Mensaje extenso " * 500}
            for i in range(40)
        ]
        compacted = engine._compact_history(huge_history, max_messages=20, max_history_chars=4000)
        assert len(compacted) <= 20
        assert sum(len(m["content"]) for m in compacted) <= 4000

        # 3. Test automatic compaction when remaining context window is below 10%
        headroom_before = engine.evaluate_context_headroom(
            system_prompt=huge_text,
            user_prompt="Pregunta actual",
            history=huge_history,
            min_free_ratio=0.10,
        )
        assert headroom_before["remaining_ratio"] < 0.10
        assert headroom_before["needs_auto_compaction"] is True

        new_sys, new_user, new_hist, was_compacted = engine.auto_compact_when_low(
            system_prompt=huge_text,
            user_prompt="Pregunta actual",
            history=huge_history,
            min_free_ratio=0.10,
        )
        assert was_compacted is True
        assert any("Resumen compacto de turnos previos (<10% ventana libre)" in m["content"] for m in new_hist)
        headroom_after = engine.evaluate_context_headroom(
            system_prompt=new_sys,
            user_prompt=new_user,
            history=new_hist,
            min_free_ratio=0.10,
        )
        assert headroom_after["remaining_ratio"] >= 0.10
        assert headroom_after["needs_auto_compaction"] is False


