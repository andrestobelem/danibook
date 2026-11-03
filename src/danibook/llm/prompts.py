from typing import Optional

SYSTEM_AUTHOR_PROMPT = """Eres un asistente inteligente de investigación, síntesis y redacción basado en fuentes locales.

CÓMO CONVERSAR Y COLABORAR:
1. Conversa de manera natural, fluida y clara, adaptándote a las preguntas, análisis o ideas del usuario.
2. Si el usuario hace una pregunta general, pide una opinión o quiere explorar una idea, responde con naturalidad y rigor, sin rigidez robótica.
3. Cuando te apoyes en los documentos cargados (de la 'MESA DE TRABAJO' o 'CITAS DE REFERENCIA'), fundamenta tus respuestas e incluye la cita entre corchetes (ej: [Documento.docx], [Archivo.pdf, Pág. 4]) de forma orgánica.
4. Si detectas matices, conexiones o contradicciones entre las fuentes, señálalos de manera constructiva.
"""

def build_chat_prompt(
    desk_prefix: str,
    reference_context: str,
    user_query: str,
    action_type: Optional[str] = None
) -> str:
    """Builds the complete prompt combining Tier 1 Desk, Tier 2 References, and specific author workflows."""

    instructions = ""
    if action_type == "contradictions":
        instructions = (
            "TAREA ESPECIAL: Analiza minuciosamente los documentos de la Mesa de Trabajo y las Citas de Referencia. "
            "Identifica contradicciones, inconsistencias cronológicas, cambios injustificados en personajes o datos que no coincidan. "
            "Cita explícitamente el origen de cada discrepancia."
        )
    elif action_type == "timeline":
        instructions = (
            "TAREA ESPECIAL: Extrae y ordena cronológicamente todos los eventos, fechas, hitos y arcos temporales "
            "mencionados en las fuentes. Presenta una línea de tiempo estructurada con citas exactas."
        )
    elif action_type == "summary":
        instructions = (
            "TAREA ESPECIAL: Genera un resumen ejecutivo de las ideas centrales, tesis o sucesos principales de las fuentes activas, "
            "organizado por temas o capítulos con citas."
        )
    elif action_type == "quotes":
        instructions = (
            "TAREA ESPECIAL: Extrae las citas textuales más valiosas e impactantes sobre el tema solicitado, "
            "manteniendo el texto exacto entre comillas e indicando la fuente y página precisa."
        )

    context_blocks = []
    if desk_prefix:
        context_blocks.append(desk_prefix)
    if reference_context:
        context_blocks.append(reference_context)

    combined_context = "\n\n".join(context_blocks)

    prompt = f"{SYSTEM_AUTHOR_PROMPT}\n\n"
    if combined_context:
        prompt += f"{combined_context}\n\n"
    if instructions:
        prompt += f"[INSTRUCCIÓN DE TRABAJO: {instructions}]\n\n"

    prompt += f"CONSULTA DEL USUARIO:\n{user_query}\n\nRESPUESTA:\n"
    return prompt
