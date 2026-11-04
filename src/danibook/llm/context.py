from typing import List, Dict, Tuple, Optional, Any
import re


class ContextBudgetManager:
    """Manages token estimation, token clamping, and automatic turn compaction.
    
    Guarantees that inference inputs + generation headroom never exceed the model context length.
    """

    def __init__(self, tokenizer_fn=None):
        self._tokenizer_fn = tokenizer_fn

    def count_tokens_safe(self, text: str) -> int:
        """Counts tokens using the tokenizer function or a conservative 2.5 chars/token fallback."""
        if not text:
            return 0
        if self._tokenizer_fn is not None:
            try:
                return self._tokenizer_fn(text)
            except Exception:
                pass
        return max(1, int(len(text) / 2.5))

    def clamp_to_token_budget(self, text: str, max_allowed_tokens: int) -> str:
        """Guarantees that a string never exceeds max_allowed_tokens."""
        if not text or max_allowed_tokens <= 0:
            return ""
        current_tokens = self.count_tokens_safe(text)
        if current_tokens <= max_allowed_tokens:
            return text
        ratio = max(0.1, (max_allowed_tokens - 32) / max(1, current_tokens))
        cut_chars = max(100, int(len(text) * ratio))
        trimmed = text[:cut_chars]
        while self.count_tokens_safe(trimmed) > max_allowed_tokens and len(trimmed) > 100:
            trimmed = trimmed[: int(len(trimmed) * 0.85)]
        return trimmed + "\n[... Recortado automáticamente para respetar la ventana de contexto ...]"

    @staticmethod
    def compact_history(
        history: List[Dict[str, str]],
        max_messages: int = 20,
        chunk_step: int = 6,
        max_history_chars: int = 12000,
    ) -> List[Dict[str, str]]:
        """Chunked eviction and character cap so history never overflows the context window."""
        if len(history) > max_messages:
            overflow = len(history) - max_messages
            blocks_to_drop = ((overflow + chunk_step - 1) // chunk_step) * chunk_step
            history = history[blocks_to_drop:]
        while len(history) > 2 and sum(len(m.get("content", "")) for m in history) > max_history_chars:
            if history[0].get("content", "").startswith("[Resumen compacto") or history[0].get("content", "").startswith("[COMPACTACIÓN"):
                history = [history[0]] + history[3:]
            else:
                history = history[2:]
        total_chars = sum(len(m.get("content", "")) for m in history)
        if total_chars > max_history_chars and history:
            per_msg_budget = max(200, max_history_chars // len(history))
            history = [
                {**m, "content": m.get("content", "")[:per_msg_budget]}
                for m in history
            ]
        return history

    def evaluate_context_headroom(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        image_uris: Optional[List[str]] = None,
        n_ctx: int = 8192,
        max_tokens: int = 2048,
        min_free_ratio: float = 0.10,
    ) -> Dict[str, Any]:
        """Calculates context window usage and reports whether free headroom is below min_free_ratio (default 10%)."""
        image_tokens = len(image_uris or []) * 800
        sys_tokens = self.count_tokens_safe(system_prompt)
        user_tokens = self.count_tokens_safe(user_prompt)
        hist_tokens = sum(self.count_tokens_safe(m.get("content", "")) for m in (history or []))
        reserved_output = min(max_tokens, max(512, n_ctx // 4))
        total_used = sys_tokens + user_tokens + hist_tokens + image_tokens + reserved_output + 128
        remaining_tokens = max(0, n_ctx - total_used)
        remaining_ratio = remaining_tokens / max(1, n_ctx)
        return {
            "n_ctx": n_ctx,
            "total_used": total_used,
            "remaining_tokens": remaining_tokens,
            "remaining_ratio": remaining_ratio,
            "needs_auto_compaction": remaining_ratio < min_free_ratio,
        }

    def auto_compact_when_low(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        image_uris: Optional[List[str]] = None,
        n_ctx: int = 8192,
        max_tokens: int = 2048,
        min_free_ratio: float = 0.10,
    ) -> tuple[str, str, List[Dict[str, str]], bool]:
        """Automatically compacts history and prompts whenever remaining context window falls below 10%."""
        hist_list = list(history or [])
        status = self.evaluate_context_headroom(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=hist_list,
            image_uris=image_uris,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            min_free_ratio=min_free_ratio,
        )
        compacted = False
        if status["needs_auto_compaction"]:
            compacted = True
            if len(hist_list) > 4:
                older = hist_list[:-4]
                recent = hist_list[-4:]
                digest_lines = [
                    f"- {'Usuario' if m.get('role') == 'user' else 'Asistente'}: {m.get('content', '')[:160].strip()}"
                    for m in older[-8:]
                ]
                digest_msg = {
                    "role": "user",
                    "content": "[Resumen compacto de turnos previos (<10% ventana libre)]:\n" + "\n".join(digest_lines),
                }
                hist_list = [digest_msg] + recent

        image_token_reserve = len(image_uris or []) * 800
        safe_max_tokens = min(max_tokens, max(512, n_ctx // 4))
        mandatory_free_headroom = int(n_ctx * min_free_ratio)
        max_input_tokens = max(
            1024,
            min(7000, n_ctx - safe_max_tokens - image_token_reserve - mandatory_free_headroom - 256),
        )

        user_budget = max(512, min(3200, int(max_input_tokens * 0.45)))
        sys_budget = max(512, min(2800, int(max_input_tokens * 0.40)))
        hist_char_budget = max(800, min(3500, int(max_input_tokens * 0.15 * 2.5)))

        new_user = self.clamp_to_token_budget(user_prompt, user_budget)
        new_sys = self.clamp_to_token_budget(system_prompt, sys_budget)
        new_hist = self.compact_history(hist_list, max_history_chars=hist_char_budget)

        if new_user != user_prompt or new_sys != system_prompt or len(new_hist) != len(history or []):
            compacted = True

        return new_sys, new_user, new_hist, compacted
