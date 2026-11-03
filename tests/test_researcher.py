from danibook.ingestion.researcher import DeepResearcher

def test_deep_researcher_offline_graceful():
    researcher = DeepResearcher(max_results=2)
    dossier = researcher.compile_research_dossier("Historia de la imprenta")
    assert "INFORME DE DEEP RESEARCH" in dossier
    assert "Fuentes Consultadas" in dossier
