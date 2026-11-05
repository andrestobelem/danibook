from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any
import gc
import re
import threading
from ..config import DEFAULT_MODEL_ID, GEMMA_MODELS, MODELS_DIR, CACHE_DIR, ModelPreset
from .prompts import NAMING_NOTE_PROMPT, NAMING_NOTEBOOK_PROMPT, SOURCES_OVERVIEW_PROMPT
from .lifecycle import ModelLifecycleManager
from .context import ContextBudgetManager


class LLMEngine:
    """Manages Gemma 4 inference with Metal acceleration, multimodal vision, and context window stability.
    
    Delegates model file management to ModelLifecycleManager and token budgeting to ContextBudgetManager.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_mock: bool = False,
        model_id: str = DEFAULT_MODEL_ID,
    ):
        self._lock = threading.Lock()
        self.use_mock = use_mock
        self.active_model_id = model_id if model_id in GEMMA_MODELS else DEFAULT_MODEL_ID
        self.preset: ModelPreset = GEMMA_MODELS[self.active_model_id]
        self.model_path = model_path or self.preset.local_path
        self._llm = None
        self._vision_enabled = False
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lifecycle = ModelLifecycleManager(MODELS_DIR)
        self.context_mgr = ContextBudgetManager(tokenizer_fn=self._tokenize_length)

        if not self.use_mock and self.model_path and self.model_path.exists():
            self._load_llama()
        elif not self.use_mock and (not self.model_path or not self.model_path.exists()):
            # If the default model file does not exist on disk, do not claim it is active
            self.active_model_id = None

    @property
    def is_model_loaded(self) -> bool:
        """Indicates whether a real LLM model is currently loaded in memory."""
        return self._llm is not None or self.use_mock

    @property
    def download_state(self) -> Dict[str, Any]:
        """Exposes download state for backward compatibility."""
        return self.lifecycle.download_state

    def _tokenize_length(self, text: str) -> int:
        if self._llm is not None and hasattr(self._llm, "tokenize"):
            return len(self._llm.tokenize(text.encode("utf-8", errors="ignore"), add_bos=False))
        return max(1, int(len(text) / 2.5))

    def unload_active_model(self):
        """Explicitly frees the active GGUF model and CLIP projector from Metal/GPU memory."""
        with self._lock:
            if self._llm is not None:
                try:
                    if hasattr(self._llm, "close"):
                        self._llm.close()
                except Exception:
                    pass
                self._llm = None
                self._vision_enabled = False
                self.active_model_id = None
                gc.collect()

    def delete_model_from_disk(self, model_id: str) -> Dict[str, Any]:
        """Deletes a model's local files via lifecycle manager."""
        return self.lifecycle.delete_model_from_disk(model_id, self.active_model_id)

    def get_models_status(self) -> List[Dict[str, Any]]:
        """Returns disk and download status of all presets."""
        return self.lifecycle.get_models_status(self.active_model_id)

    def switch_model(self, model_id: str, auto_download: bool = True) -> Dict[str, Any]:
        """Switches the active model, downloading from Hugging Face if needed and loading into Metal."""
        if model_id not in GEMMA_MODELS:
            raise ValueError(f"Modelo desconocido: {model_id}")

        preset = GEMMA_MODELS[model_id]

        if not preset.is_downloaded and not auto_download:
            raise FileNotFoundError(f"El modelo {preset.name} no está en disco.")

        if not preset.is_downloaded or (preset.supports_vision and not preset.is_vision_ready):
            self.lifecycle.download_model_files(preset)

        self.unload_active_model()

        with self._lock:
            self.active_model_id = model_id
            self.preset = preset
            self.model_path = preset.local_path

            if not self.use_mock:
                self._load_llama()

        return {
            "model_id": self.active_model_id,
            "name": self.preset.name,
            "context_length": self.preset.context_length,
            "supports_vision": self.preset.supports_vision,
            "vision_enabled": self._vision_enabled,
        }

    def _load_llama(self):
        """Loads llama.cpp with Apple Silicon Metal offloading and optional vision clip projector."""
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Gemma4ChatHandler

            chat_handler = None
            self._vision_enabled = False
            if self.preset.supports_vision and self.preset.mmproj_local_path and self.preset.mmproj_local_path.exists():
                try:
                    chat_handler = Gemma4ChatHandler(clip_model_path=str(self.preset.mmproj_local_path))
                    self._vision_enabled = True
                except Exception:
                    chat_handler = None
                    self._vision_enabled = False

            self._llm = Llama(
                model_path=str(self.model_path),
                chat_handler=chat_handler,
                n_ctx=self.preset.context_length,
                n_batch=512,
                n_gpu_layers=-1,  # Offload to Metal
                verbose=False,
            )
        except ImportError:
            self.use_mock = True

    # Delegate token & context management methods to maintain exact API compatibility
    def count_tokens_safe(self, text: str) -> int:
        return self.context_mgr.count_tokens_safe(text)

    def clamp_to_token_budget(self, text: str, max_allowed_tokens: int) -> str:
        return self.context_mgr.clamp_to_token_budget(text, max_allowed_tokens)

    def _compact_history(self, *args, **kwargs):
        """Preserves backward compatibility alias for history compaction."""
        return self.context_mgr.compact_history(*args, **kwargs)

    def evaluate_context_headroom(self, *args, **kwargs):
        if "n_ctx" not in kwargs and (len(args) < 5):
            kwargs["n_ctx"] = self.preset.context_length
        return self.context_mgr.evaluate_context_headroom(*args, **kwargs)

    def auto_compact_when_low(self, *args, **kwargs):
        if "n_ctx" not in kwargs and (len(args) < 5):
            kwargs["n_ctx"] = self.preset.context_length
        return self.context_mgr.auto_compact_when_low(*args, **kwargs)

    def generate_chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        image_uris: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """Generates a streaming response, respecting context bounds and multimodal inputs."""
        if self._llm is None and not self.use_mock:
            yield "⚠️ **No model is currently loaded in memory.**\n\n"
            yield "Please click the **[Models]** button in the top bar to download or activate a Gemma model from Hugging Face before asking questions."
            return

        if self.use_mock:
            yield from self._mock_stream(user_prompt)
            return

        with self._lock:
            n_ctx = self.preset.context_length
            reserved_vision_tokens = 1500 if (self._vision_enabled and image_uris) else 0
            safe_max_tokens = min(2048, max(512, n_ctx // 4))

            # Auto-compact conversation turns if free context ratio drops below 10%
            safe_system, user_prompt, compacted_history, _ = self.context_mgr.auto_compact_when_low(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                history=history,
                image_uris=image_uris,
                n_ctx=n_ctx,
                max_tokens=safe_max_tokens,
                min_free_ratio=0.10,
            )

            # Cap total history
            safe_history = self.context_mgr.compact_history(
                compacted_history,
                max_messages=14 if n_ctx <= 8192 else 24,
                max_history_chars=4000 if n_ctx <= 8192 else 14000,
            )

            # Clamp prompts to context budget
            remaining_for_prompts = max(
                512,
                n_ctx - (safe_max_tokens + reserved_vision_tokens + sum(self.count_tokens_safe(m.get("content", "")) for m in safe_history))
            )
            safe_system = self.clamp_to_token_budget(system_prompt, int(remaining_for_prompts * 0.70))
            safe_user = self.clamp_to_token_budget(user_prompt, int(remaining_for_prompts * 0.30))

            messages: List[Dict[str, Any]] = []
            if safe_system:
                messages.append({"role": "system", "content": safe_system})

            for turn in safe_history:
                messages.append({"role": turn["role"], "content": turn.get("content", "")})

            def format_user_content(text: str, include_images: bool = False) -> Any:
                if not include_images or not image_uris or not self._vision_enabled:
                    return text
                content_parts = [{"type": "text", "text": text}]
                for uri in image_uris:
                    content_parts.append({"type": "image_url", "image_url": {"url": uri}})
                return content_parts

            user_payload = format_user_content(safe_user, include_images=True)
            messages.append({"role": "user", "content": user_payload})

            try:
                response = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=safe_max_tokens,
                    stream=True,
                    temperature=0.7,
                )
                for chunk in response:
                    text = chunk["choices"][0].get("delta", {}).get("content", "")
                    if text:
                        yield text
            except ValueError as ve:
                if "exceed context window" in str(ve) or "exceeds n_ctx" in str(ve):
                    yield "⚠️ *Ajustando recorte automático de contexto...*\n\n"
                    emergency_budget = max(512, (n_ctx - safe_max_tokens) // 3)
                    fallback_prompt = self.clamp_to_token_budget(
                        f"{safe_system}\n\n{safe_user}" if safe_system else safe_user,
                        emergency_budget,
                    )
                    retry_resp = self._llm.create_chat_completion(
                        messages=[{"role": "user", "content": fallback_prompt}],
                        max_tokens=safe_max_tokens,
                        stream=True,
                        temperature=0.7,
                    )
                    for chunk in retry_resp:
                        text = chunk["choices"][0].get("delta", {}).get("content", "")
                        if text:
                            yield text
                else:
                    yield f"\n\n⚠️ Error de inferencia: {ve}"
            except Exception as exc:
                yield f"\n\n⚠️ Error inesperado en el modelo: {exc}"

    def _mock_stream(self, prompt: str) -> Iterator[str]:
        sample_response = (
            "I analyzed the documents in your active desk.\n\n"
            "Regarding your query, [Capitulo_1] states a central premise that connects "
            "directly to the historical records in [Investigacion.epub, Page 4].\n\n"
            "<!-- SUGGESTED_PROMPTS_START -->\n"
            "- How does this thesis compare with other active sources?\n"
            "- Can you extract key quotes supporting this argument?\n"
            "- Would you like a timeline of these events?\n"
            "- What are the potential counterarguments?\n"
            "<!-- SUGGESTED_PROMPTS_END -->"
        )
        for word in sample_response.split(" "):
            yield word + " "

    def generate_note_title(self, text: str) -> str:
        """Generates a concise 2-6 word title for a note using the active model or fallback."""
        if not text or not text.strip():
            return "Note"
        if self.use_mock or self._llm is None:
            words = [w for w in re.sub(r"[^\w\s]", "", text).split() if len(w) > 2]
            return " ".join(words[:4]).title() if words else "Research Note"

        prompt = NAMING_NOTE_PROMPT.format(text=text[:1200])
        with self._lock:
            try:
                res = self._llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=24,
                    temperature=0.3,
                )
                raw_title = res["choices"][0]["message"].get("content", "").strip()
                cleaned = re.sub(r'[\r\n"\'`.]+', '', raw_title).strip()
                words = cleaned.split()
                if 1 <= len(words) <= 8:
                    return " ".join(words)
            except Exception:
                pass
        words = [w for w in re.sub(r"[^\w\s]", "", text).split() if len(w) > 2]
        return " ".join(words[:4]).title() if words else "Research Note"

    def generate_notebook_overview(self, sources_summary: str) -> Dict[str, str]:
        """Generates a short overview of the sources and a concise notebook title."""
        if not sources_summary or not sources_summary.strip():
            return {
                "title": "Research Notebook",
                "overview": "No sources selected. Upload or select documents to begin analysis."
            }

        default_title = "Research Notebook"
        default_overview = "Sources loaded and ready for analysis. Ask questions or run guided actions below."

        if self.use_mock or self._llm is None:
            return {
                "title": default_title,
                "overview": default_overview
            }

        overview = default_overview
        title = default_title

        with self._lock:
            try:
                ov_resp = self._llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a concise research assistant. Follow ASD-STE100 principles."},
                        {"role": "user", "content": f"{SOURCES_OVERVIEW_PROMPT}\n\nSources:\n{sources_summary[:3000]}"}
                    ],
                    max_tokens=180,
                    temperature=0.3,
                )
                raw_ov = ov_resp["choices"][0]["message"].get("content", "").strip()
                if raw_ov:
                    overview = raw_ov
            except Exception:
                pass

            try:
                title_resp = self._llm.create_chat_completion(
                    messages=[
                        {"role": "user", "content": NAMING_NOTEBOOK_PROMPT.format(text=overview)}
                    ],
                    max_tokens=24,
                    temperature=0.3,
                )
                raw_title = title_resp["choices"][0]["message"].get("content", "").strip()
                cleaned = re.sub(r'[\r\n"\'`.]+', '', raw_title).strip()
                if cleaned:
                    title = cleaned
            except Exception:
                pass

        return {
            "title": title,
            "overview": overview
        }
