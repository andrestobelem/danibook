from typing import Optional, List, Dict, Tuple
from ..memory.tier1_desk import DeskMemory
from ..memory.tier2_archive import ArchiveMemory
from ..llm.prompts import SYSTEM_AUTHOR_PROMPT, build_chat_prompt


class StudioSynthesizer:
    """Calculates token budgets and prepares prompts for Studio chat and artifact synthesis."""

    @staticmethod
    def calculate_budgets(context_length: int) -> Tuple[int, int]:
        """Calculates character budgets for Desk prefix and Archive search chunks based on context length."""
        desk_char_budget = max(2500, min(14000, int((context_length - 5000) * 1.5)))
        archive_chunk_budget = 900 if context_length <= 8192 else 2200
        return desk_char_budget, archive_chunk_budget

    @classmethod
    def assemble_prompt(
        cls,
        desk: DeskMemory,
        archive: ArchiveMemory,
        message: str,
        action_type: str = "chat",
        context_length: int = 8192,
        image_uri: Optional[str] = None,
    ) -> Tuple[str, str, List[str]]:
        """Assembles the system prompt, user prompt, and active image URIs."""
        desk_char_budget, archive_chunk_budget = cls.calculate_budgets(context_length)

        desk_prefix = desk.build_desk_prefix(max_chars=desk_char_budget)
        active_images = desk.get_active_image_uris(max_images=2)
        if image_uri and image_uri not in active_images:
            active_images.insert(0, image_uri)

        reference_context = archive.format_search_context(
            message, top_k=3, max_chunk_chars=archive_chunk_budget
        )

        base_system_context = SYSTEM_AUTHOR_PROMPT
        if desk_prefix:
            base_system_context += f"\n\n{desk_prefix}"

        current_turn_prompt = build_chat_prompt(
            desk_prefix="",
            reference_context=reference_context,
            user_query=message,
            action_type=action_type,
        )

        return base_system_context, current_turn_prompt, active_images
