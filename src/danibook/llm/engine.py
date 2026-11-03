from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any
import os
import gc
import urllib.request
from ..config import DEFAULT_MODEL_ID, GEMMA_MODELS, MODELS_DIR, CACHE_DIR, ModelPreset


class LLMEngine:
    """Manages Gemma 4 inference with Hugging Face on-demand downloads, multimodal vision (Gemma4ChatHandler), and KV prefix stability."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_mock: bool = False,
        model_id: str = DEFAULT_MODEL_ID,
    ):
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
        req = urllib.request.Request(url, headers={"User-Agent": "Danibook/1.0"})
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

    @staticmethod
    def _compact_history(history: List[Dict[str, str]], max_messages: int = 20, chunk_step: int = 6) -> List[Dict[str, str]]:
        """Chunked eviction so KV prefix does not shift on every single turn after reaching the limit."""
        if len(history) <= max_messages:
            return history
        overflow = len(history) - max_messages
        blocks_to_drop = ((overflow + chunk_step - 1) // chunk_step) * chunk_step
        return history[blocks_to_drop:]

    def generate_chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        image_uris: Optional[List[str]] = None,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """Streams response tokens preserving multi-turn conversation history and optional multimodal images."""
        if self.use_mock or self._llm is None:
            yield from self._mock_stream(user_prompt)
            return

        messages: List[Dict[str, Any]] = []
        history = self._compact_history(history or [])
        active_images = (image_uris or []) if self._vision_enabled else []

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

        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
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
                yield "⚠️ *El volumen de fuentes activas excedió la ventana inmediata. Ajustando recorte de contexto...*\n\n"
                fallback_prompt = (system_prompt[:12000] + "\n\n" + user_prompt[:8000]) if system_prompt else user_prompt[:12000]
                retry_resp = self._llm.create_chat_completion(
                    messages=[{"role": "user", "content": fallback_prompt}],
                    max_tokens=max_tokens,
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
            "¡Hola! Estuve analizando los documentos que cargaste en tu mesa de trabajo.\n\n"
            "Respecto a tu consulta, en el [Capítulo 1] se plantea una premisa clave que conecta "
            "directamente con los datos registrados en [Investigacion.epub, Pág. 4].\n\n"
            "¿Querés que profundicemos en algún punto específico o que comparemos otra sección?"
        )
        for word in sample_response.split(" "):
            yield word + " "
