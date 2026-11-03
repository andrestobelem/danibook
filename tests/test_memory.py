import pytest
from pathlib import Path
from danibook.ingestion.parser import DocumentParser
from danibook.memory.tier1_desk import DeskMemory
from danibook.memory.tier2_archive import ArchiveMemory

def test_tier1_desk_memory(tmp_path):
    f1 = tmp_path / "cap1.md"
    f1.write_text("Texto del primer capítulo.", encoding="utf-8")
    doc1 = DocumentParser.parse(f1)

    desk = DeskMemory()
    desk.add_document(doc1)
    
    assert "cap1.md" in desk.get_active_files()
    prefix = desk.build_desk_prefix()
    assert "MESA DE TRABAJO" in prefix
    assert "Texto del primer capítulo." in prefix

def test_tier2_archive_incremental_search(tmp_path):
    f_ref = tmp_path / "investigacion.md"
    f_ref.write_text("En 1942 ocurrió un evento histórico en Europa.", encoding="utf-8")
    doc_ref = DocumentParser.parse(f_ref)

    archive = ArchiveMemory(db_path=tmp_path / "test_fts5.sqlite")
    archive.add_document(doc_ref)

    # Search relevant query
    results = archive.search("evento histórico 1942")
    assert len(results) == 1
    assert "1942" in results[0][0].content

    # Search irrelevant query
    irrelevant = archive.search("física cuántica átomos")
    assert len(irrelevant) == 0
