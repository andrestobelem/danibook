from pathlib import Path
from typing import Dict, Any, Optional
import os
import gc
import threading
import time
import urllib.request
from ..config import DEFAULT_MODEL_ID, GEMMA_MODELS, MODELS_DIR, ModelPreset


class ModelLifecycleManager:
    """Manages model catalog state, on-demand GGUF downloads from Hugging Face, and disk cleanup.
    
    Decoupled from inference execution to ensure high cohesion and single responsibility.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.download_state: Dict[str, Any] = {
            "active": False,
            "model_id": None,
            "percent": 0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "status_text": "",
        }

    def get_models_status(self, active_model_id: str) -> list[Dict[str, Any]]:
        """Returns the disk and download status of all configured Gemma model presets."""
        status_list = []
        for mid, preset in GEMMA_MODELS.items():
            is_active = mid == active_model_id
            local_exists = preset.is_downloaded
            vision_ready = preset.is_vision_ready if preset.supports_vision else False

            status_list.append({
                "id": preset.id,
                "name": preset.name,
                "repo_id": preset.repo_id,
                "size_label": preset.size_label,
                "context_length": preset.context_length,
                "supports_vision": preset.supports_vision,
                "is_downloaded": local_exists,
                "downloaded": local_exists,
                "is_vision_ready": vision_ready,
                "is_active": is_active,
                "active": is_active,
                "size_on_disk_mb": round(preset.local_path.stat().st_size / (1024 * 1024), 1) if local_exists else 0.0,
                "is_downloading": (
                    self.download_state["active"]
                    and self.download_state["model_id"] == mid
                ),
            })
        return status_list

    def delete_model_from_disk(self, model_id: str, active_model_id: str) -> Dict[str, Any]:
        """Deletes the GGUF and mmproj files for the given model if it is not currently active."""
        if model_id not in GEMMA_MODELS:
            raise ValueError(f"Modelo desconocido: {model_id}")
        if model_id == active_model_id:
            raise ValueError(
                "No puedes borrar el modelo actualmente cargado en memoria. "
                "Cambia a otro modelo antes de eliminar este del disco."
            )

        preset = GEMMA_MODELS[model_id]
        deleted_files = []
        freed_bytes = 0

        if preset.local_path.exists():
            freed_bytes += preset.local_path.stat().st_size
            preset.local_path.unlink()
            deleted_files.append(preset.local_path.name)

        if preset.mmproj_local_path and preset.mmproj_local_path.exists():
            freed_bytes += preset.mmproj_local_path.stat().st_size
            preset.mmproj_local_path.unlink()
            deleted_files.append(preset.mmproj_local_path.name)

        return {
            "deleted": True,
            "model_id": model_id,
            "files": deleted_files,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
            "models": self.get_models_status(active_model_id),
        }

    def download_model_files(self, preset: ModelPreset):
        """Downloads the main GGUF and optional mmproj for a model preset synchronously."""
        if not preset.local_path.exists():
            self._download_file_with_progress(
                preset.repo_id,
                preset.hf_filename,
                preset.local_path,
                preset.id,
                f"Downloading {preset.name}...",
            )

        if preset.supports_vision and preset.mmproj_hf_filename and preset.mmproj_local_path:
            if not preset.mmproj_local_path.exists():
                self._download_file_with_progress(
                    preset.mmproj_repo_id or preset.repo_id,
                    preset.mmproj_hf_filename,
                    preset.mmproj_local_path,
                    preset.id,
                    f"Downloading visual CLIP projector ({preset.name})...",
                )

    def _download_file_with_progress(
        self,
        repo_id: str,
        filename: str,
        dest_path: Path,
        model_id: str,
        label: str,
    ):
        """Downloads a file from Hugging Face with thread-safe percentage, speed, and MB reporting."""
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = dest_path.with_suffix(dest_path.suffix + ".downloading")

        with self._lock:
            self.download_state["active"] = True
            self.download_state["model_id"] = model_id
            self.download_state["percent"] = 0
            self.download_state["downloaded_mb"] = 0.0
            self.download_state["total_mb"] = 0.0
            self.download_state["speed_mb_s"] = 0.0
            self.download_state["status_text"] = label

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DaniBookLM/0.2"})
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                total_mb = round(total_size / (1024 * 1024), 1)
                with self._lock:
                    self.download_state["total_mb"] = total_mb

                downloaded = 0
                block_size = 512 * 1024  # 512 KB blocks for smooth progress updates
                start_time = time.time()
                last_update_time = start_time

                with open(temp_dest, "wb") as f:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        downloaded += len(buffer)
                        now = time.time()
                        
                        # Throttle state lock updates to at most every 0.1s
                        if now - last_update_time >= 0.1 or downloaded == total_size:
                            elapsed = max(0.01, now - start_time)
                            speed_mb = round((downloaded / (1024 * 1024)) / elapsed, 2)
                            pct = int((downloaded / total_size) * 100) if total_size else 50
                            downloaded_mb = round(downloaded / (1024 * 1024), 1)
                            
                            with self._lock:
                                self.download_state["percent"] = min(99, pct)
                                self.download_state["downloaded_mb"] = downloaded_mb
                                self.download_state["speed_mb_s"] = speed_mb
                                self.download_state["status_text"] = f"{label} ({downloaded_mb} / {total_mb} MB • {speed_mb} MB/s)"
                            last_update_time = now

            temp_dest.rename(dest_path)
            with self._lock:
                self.download_state["percent"] = 100
                self.download_state["status_text"] = f"✓ {filename} download complete"
        except Exception as e:
            if temp_dest.exists():
                temp_dest.unlink()
            raise RuntimeError(f"Error downloading {filename} from Hugging Face: {e}")
        finally:
            with self._lock:
                self.download_state["active"] = False
