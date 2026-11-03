from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any
import os
import gc
import re
import threading
import urllib.request
from ..config import DEFAULT_MODEL_ID, GEMMA_MODELS, MODELS_DIR, CACHE_DIR, ModelPreset
from .prompts import NAMING_NOTE_PROMPT, NAMING_NOTEBOOK_PROMPT, SOURCES_OVERVIEW_PROMPT


class LLMEngine:
    """Manages Gemma 4 inference with Hugging Face on-demand downloads, multimodal vision (Gemma4ChatHandler), and KV prefix stability."""

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
        self.download_state: Dict[str, Any] = {
            "active": False,
            "model_id": None,
            "percent": 0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "status_text": "",
        }

        if not self.use_mock and self.model_path and self.model_path.exists():
            self._load_llama()

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
            gc.collect()

    def delete_model_from_disk(self, model_id: str) -> Dict[str, Any]:
        """Deletes a model GGUF file (and optional mmproj) from disk. If it was active, unloads it first."""
        if model_id not in GEMMA_MODELS:
            raise ValueError(f"Unknown model_id: {model_id}")
        preset = GEMMA_MODELS[model_id]
        if self.active_model_id == model_id:
            self.unload_active_model()

        deleted_files = []
        if preset.local_path.exists():
            preset.local_path.unlink()
            deleted_files.append(preset.local_path.name)
        if preset.mmproj_local_path and preset.mmproj_local_path.exists():
            preset.mmproj_local_path.unlink()
            deleted_files.append(preset.mmproj_local_path.name)

        return {
            "deleted_model_id": model_id,
            "deleted_files": deleted_files,
            "models": self.get_models_status(),
        }

    def _download_file_with_progress(self, repo_id: str, filename: str, dest_path: Path, model_id: str, label: str):
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        temp_path = dest_path.with_suffix(".part")
        self.download_state.update({
            "active": True,
            "model_id": model_id,
            "percent": 0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "status_text": f"Descargando {label}...",
        })
        req = urllib.request.Request(url, headers={"User-Agent": "DaniBookLM/1.0"})
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_f:
            total_bytes = int(resp.info().get("Content-Length") or 0)
            total_mb = total_bytes / (1024 * 1024) if total_bytes else 0.0
            self.download_state["total_mb"] = round(total_mb, 1)
            downloaded = 0
            chunk_size = 1024 * 512  # 512 KB chunks
            while True:
                buf = resp.read(chunk_size)
                if not buf:
                    break
                out_f.write(buf)
                downloaded += len(buf)
                dl_mb = downloaded / (1024 * 1024)
                pct = int((downloaded * 100) / total_bytes) if total_bytes else 50
                self.download_state.update({
                    "percent": min(99, pct),
                    "downloaded_mb": round(dl_mb, 1),
                    "status_text": f"Descargando {label}: {dl_mb:.1f} MB / {total_mb:.1f} MB ({pct}%)",
                })
        temp_path.replace(dest_path)

    def get_models_status(self) -> List[Dict[str, Any]]:
        result = []
        for mid, preset in GEMMA_MODELS.items():
            size_on_disk_mb = 0.0
            if preset.local_path.exists():
                size_on_disk_mb += preset.local_path.stat().st_size / (1024 * 1024)
            if preset.mmproj_local_path and preset.mmproj_local_path.exists():
                size_on_disk_mb += preset.mmproj_local_path.stat().st_size / (1024 * 1024)
            result.append({
                "id": mid,
                "name": preset.name,
                "repo_id": preset.repo_id,
                "size_label": preset.size_label,
                "size_on_disk_mb": round(size_on_disk_mb, 1),
                "downloaded": preset.is_downloaded,
                "supports_vision": preset.supports_vision,
                "vision_ready": preset.is_vision_ready,
                "context_length": preset.context_length,
                "active": (mid == self.active_model_id and self._llm is not None),
            })
        return result

    def switch_model(self, model_id: str, auto_download: bool = True) -> Dict[str, Any]:
        """Unloads current model from memory and loads/downloads the selected Gemma model."""
        if model_id not in GEMMA_MODELS:
            raise ValueError(f"Unknown model_id: {model_id}")

        # 1. Always unload previous model from Metal/VRAM first
        self.unload_active_model()

        preset = GEMMA_MODELS[model_id]
        try:
            if not preset.is_downloaded and auto_download:
                self._download_file_with_progress(
                    repo_id=preset.repo_id,
                    filename=preset.hf_filename,
                    dest_path=preset.local_path,
                    model_id=model_id,
                    label=preset.name,
                )

            if preset.supports_vision and preset.mmproj_local_path and not preset.mmproj_local_path.exists() and auto_download:
                try:
                    self._download_file_with_progress(
                        repo_id=preset.mmproj_repo_id,
                        filename=preset.mmproj_hf_filename,
                        dest_path=preset.mmproj_local_path,
                        model_id=model_id,
                        label=f"Proyector Visión ({preset.id})",
                    )
                except Exception:
                    pass

            self.download_state.update({
                "active": True,
                "model_id": model_id,
                "percent": 100,
                "status_text": f"Cargando {preset.name} en memoria GPU Metal...",
            })
            self.active_model_id = model_id
            self.preset = preset
            self.model_path = preset.local_path
            self.use_mock = False
            self._load_llama()
        finally:
            self.download_state.update({
                "active": False,
                "model_id": None,
                "percent": 100,
                "status_text": "Listo",
            })

        return {
            "active_model_id": self.active_model_id,
            "name": self.preset.name,
            "vision_enabled": self._vision_enabled,
        }

    def _load_llama(self):
        try:
            from llama_cpp import Llama
            chat_handler = None
            self._vision_enabled = False

            if self.preset.is_vision_ready and self.preset.mmproj_local_path:
                try:
                    from llama_cpp.llama_chat_format import Gemma4ChatHandler
                    chat_handler = Gemma4ChatHandler(
                        clip_model_path=str(self.preset.mmproj_local_path),
                        verbose=False,
                    )
                    self._vision_enabled = True
                except Exception:
                    chat_handler = None
                    self._vision_enabled = False

            self._llm = Llama(
                model_path=str(self.model_path),
                chat_handler=chat_handler,
                n_ctx=self.preset.context_length,
                n_gpu_layers=-1,  # Offload to Metal (Mac) or CUDA (Windows)
                verbose=False,
            )
        except ImportError:
            self.use_mock = True

    def count_tokens_safe(self, text: str) -> int:
        """Counts tokens using the loaded GGUF tokenizer or a conservative 2.5 chars/token fallback."""
        if not text:
            return 0
        if self._llm is not None and hasattr(self._llm, "tokenize"):
            try:
                return len(self._llm.tokenize(text.encode("utf-8", errors="ignore"), add_bos=False))
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
    def _compact_history(
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
            if history[0].get("content", "").startswith("[Resumen compacto"):
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
        max_tokens: int = 2048,
        min_free_ratio: float = 0.10,
    ) -> Dict[str, Any]:
        """Calculates context window usage and reports whether free headroom is below min_free_ratio (default 10%)."""
        n_ctx = self.preset.context_length
        active_images = (image_uris or []) if self._vision_enabled else []
        image_tokens = len(active_images) * 800
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
            max_tokens=max_tokens,
            min_free_ratio=min_free_ratio,
        )
        compacted = False
        if status["needs_auto_compaction"]:
            compacted = True
            # Compress older conversation turns into a concise digest and keep the most recent 4 turns
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

        # Guarantee at least min_free_ratio (10%) of n_ctx remains free on top of output & vision reserves
        n_ctx = self.preset.context_length
        active_images = (image_uris or []) if self._vision_enabled else []
        image_token_reserve = len(active_images) * 800
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
        new_hist = self._compact_history(hist_list, max_history_chars=hist_char_budget)

        if new_user != user_prompt or new_sys != system_prompt or len(new_hist) != len(history or []):
            compacted = True

        return new_sys, new_user, new_hist, compacted

    def generate_chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        image_uris: Optional[List[str]] = None,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """Streams response tokens while strictly enforcing the model's context_length budget and <10% auto-compaction."""
        if self.use_mock or self._llm is None:
            yield from self._mock_stream(user_prompt)
            return

        active_images = (image_uris or []) if self._vision_enabled else []
        n_ctx = self.preset.context_length
        safe_max_tokens = min(max_tokens, max(512, n_ctx // 4))

        system_prompt, user_prompt, history, _ = self.auto_compact_when_low(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
            image_uris=active_images,
            max_tokens=safe_max_tokens,
            min_free_ratio=0.10,
        )

        messages: List[Dict[str, Any]] = []

        def format_user_content(text: str, include_images: bool = False) -> Any:
            if include_images and active_images:
                parts: List[Dict[str, Any]] = [
                    {"type": "image_url", "image_url": {"url": uri}} for uri in active_images
                ]
                parts.append({"type": "text", "text": text})
                return parts
            return text

        if history:
            first_msg = history[0]
            first_content = f"{system_prompt}\n\n{first_msg['content']}" if system_prompt else first_msg["content"]
            messages.append({"role": "user", "content": first_content})

            for msg in history[1:]:
                role = "assistant" if msg.get("role") == "assistant" else "user"
                if messages and messages[-1]["role"] == role and isinstance(messages[-1]["content"], str):
                    messages[-1]["content"] += f"\n\n{msg.get('content', '')}"
                else:
                    messages.append({"role": role, "content": msg.get("content", "")})

            if messages[-1]["role"] == "user" and not active_images and isinstance(messages[-1]["content"], str):
                messages[-1]["content"] += f"\n\n{user_prompt}"
            else:
                if messages[-1]["role"] == "user":
                    messages.append({"role": "assistant", "content": "Entendido."})
                messages.append({
                    "role": "user",
                    "content": format_user_content(user_prompt, include_images=True),
                })
        else:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            messages.append({
                "role": "user",
                "content": format_user_content(combined_prompt, include_images=True),
            })

        with self._lock:
            try:
                response = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=safe_max_tokens,
                    stream=True,
                    temperature=0.7,
                    top_p=0.95,
                )

                for chunk in response:
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if not text:
                        continue

                    if "<|channel>thought" in text or "<channel|>" in text:
                        text = text.replace("<|channel>thought", "").replace("<channel|>", "").strip()

                    if text:
                        yield text
            except ValueError as ve:
                if "exceed context window" in str(ve) or "exceeds n_ctx" in str(ve):
                    yield "⚠️ *Ajustando recorte automático de contexto...*\n\n"
                    emergency_budget = max(512, (n_ctx - safe_max_tokens) // 3)
                    fallback_prompt = self.clamp_to_token_budget(
                        f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt,
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
                "title": "Cuaderno de Investigación",
                "overview": "No hay fuentes seleccionadas. Sube o selecciona documentos para comenzar el análisis."
            }

        default_title = "Cuaderno de Investigación"
        default_overview = "Fuentes cargadas y listas para consulta. Formula preguntas o ejecuta acciones guiadas en el panel inferior."

        if self.use_mock or self._llm is None:
            return {
                "title": default_title,
                "overview": default_overview
            }

        overview = default_overview
        title = default_title

        with self._lock:
            # 1. Generate overview
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

            # 2. Generate title based on overview
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
