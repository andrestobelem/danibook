from danibook.llm.prompts import build_chat_prompt

def test_build_chat_prompt():
    desk = "MESA DE TRABAJO: Capítulo 1"
    ref = "CITAS: [Investigacion.pdf, Pág 5]"
    query = "¿Quién es el personaje?"

    prompt = build_chat_prompt(desk, ref, query, action_type="contradictions")
    assert "MESA DE TRABAJO: Capítulo 1" in prompt
    assert "[Investigacion.pdf, Pág 5]" in prompt
    assert "SPECIAL TASK: Carefully analyze" in prompt
    assert "¿Quién es el personaje?" in prompt

    glossary_prompt = build_chat_prompt(desk, ref, "Armar glosario", action_type="glossary")
    assert "Build a conceptual map and glossary" in glossary_prompt

    mindmap_prompt = build_chat_prompt(desk, ref, "Create mindmap", action_type="mindmap")
    assert "SPECIAL TASK: Create a visual, structured Concept & Entity Map" in mindmap_prompt
    assert "mermaid" in mindmap_prompt

    flashcards_prompt = build_chat_prompt(desk, ref, "Generate flashcards", action_type="flashcards")
    assert "SPECIAL TASK: Generate a set of 5 to 8 high-impact study flashcards" in flashcards_prompt
