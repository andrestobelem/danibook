from danibook.llm.prompts import build_chat_prompt

def test_build_chat_prompt():
    desk = "MESA DE TRABAJO: Capítulo 1"
    ref = "CITAS: [Investigacion.pdf, Pág 5]"
    query = "¿Quién es el personaje?"

    prompt = build_chat_prompt(desk, ref, query, action_type="contradictions")
    assert "MESA DE TRABAJO: Capítulo 1" in prompt
    assert "[Investigacion.pdf, Pág 5]" in prompt
    assert "TAREA ESPECIAL: Analiza minuciosamente" in prompt
    assert "¿Quién es el personaje?" in prompt
