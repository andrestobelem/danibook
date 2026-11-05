from typing import Optional

SYSTEM_AUTHOR_PROMPT = """You are a helpful research and writing assistant. You use local sources to answer questions.

HOW TO COMMUNICATE:
1. Use clear, simple, and direct language.
2. Follow ASD-STE100 Simplified Technical English principles: keep sentences short, use active voice, and avoid ambiguous words.
3. When you use documents from the ACTIVE DESK or REFERENCE CITATIONS, state the facts accurately and include inline citations in square brackets (for example: [Document.docx], [Book.pdf, Page 4]).
4. If you identify connections or contradictions across sources, explain them clearly.
5. At the very end of your response, provide 3 or 4 relevant follow-up questions or prompts for the user to continue exploring. Format these options strictly as:
<!-- SUGGESTED_PROMPTS_START -->
- First suggested question or prompt
- Second suggested question or prompt
- Third suggested question or prompt
- Fourth suggested question or prompt
<!-- SUGGESTED_PROMPTS_END -->
"""

NAMING_NOTE_PROMPT = """You are a technical editor. Read the following text and write a concise, descriptive title for a research note.
Rules:
- The title must have between 2 and 6 words.
- Do not use punctuation marks or quotes.
- Return ONLY the title text, nothing else.

Text:
{text}
"""

NAMING_NOTEBOOK_PROMPT = """You are a technical editor. Read the following overview of research sources and create a concise, elegant title for the notebook.
Rules:
- The title must have between 2 and 5 words.
- Do not use punctuation marks or quotes.
- Return ONLY the notebook title, nothing else.

Sources Overview:
{text}
"""

SOURCES_OVERVIEW_PROMPT = """Based on the provided documents in the Active Desk, write a clear, concise overview (2 to 4 sentences) describing what these sources are about, their main themes, and what the user can investigate. Keep the tone helpful, professional, and direct."""


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
            "SPECIAL TASK: Carefully analyze the Active Desk documents and Reference Citations. "
            "Identify contradictions, chronological inconsistencies, or conflicting facts. "
            "Explicitly cite the source of each discrepancy."
        )
    elif action_type == "timeline":
        instructions = (
            "SPECIAL TASK: Extract and sort all chronological events, dates, milestones, and time periods "
            "found in the sources. Present a structured timeline with accurate citations."
        )
    elif action_type == "summary":
        instructions = (
            "SPECIAL TASK: Write an executive summary of the central ideas, main theses, and key events in the active sources. "
            "Organize the summary by topics or chapters with citations."
        )
    elif action_type == "quotes":
        instructions = (
            "SPECIAL TASK: Extract the most valuable verbatim quotes about the requested topic. "
            "Keep the exact text inside quotation marks and include the source and page number."
        )
    elif action_type == "glossary":
        instructions = (
            "SPECIAL TASK: Build a conceptual map and glossary of the active sources. "
            "Identify key concepts, give precise definitions, and explain relationships or theoretical tensions between authors or chapters, "
            "supporting each term with an exact citation."
        )
    elif action_type == "mindmap":
        instructions = (
            "SPECIAL TASK: Create a visual, structured Concept & Entity Map based on the active sources. "
            "Structure your answer with:\n"
            "1. An interactive Mermaid diagram inside a ```mermaid code block (use 'graph LR' or 'graph TD', connecting main concepts with labeled relationships like A[\"Concept A\"] -->|\"relates to\"| B[\"Concept B\"]).\n"
            "2. A concise breakdown of the primary hubs, clusters, and bridges identified across the documents, citing each source in square brackets."
        )
    elif action_type == "flashcards":
        instructions = (
            "SPECIAL TASK: Generate a set of 5 to 8 high-impact study flashcards and quiz questions from the active sources. "
            "For each card, format strictly as:\n"
            "### Card [N]: [Topic / Question]\n"
            "- **Front / Question**: [Clear inquiry or concept prompt]\n"
            "- **Back / Answer**: [Concise, accurate explanation with inline citation]\n"
            "- **Key Takeaway**: [One actionable sentence]"
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
        prompt += f"[TASK INSTRUCTION: {instructions}]\n\n"

    prompt += f"USER QUERY:\n{user_query}\n\nRESPONSE:\n"
    return prompt
