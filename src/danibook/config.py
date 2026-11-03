from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default directories for Danibook
DEFAULT_DATA_DIR = Path(os.getenv("DANIBOOK_DATA_DIR", PROJECT_ROOT / "data"))
WORKSPACE_DIR = DEFAULT_DATA_DIR / "workspace"
CACHE_DIR = DEFAULT_DATA_DIR / "cache"
MODELS_DIR = DEFAULT_DATA_DIR / "models"
NOTES_DIR = DEFAULT_DATA_DIR / "notes"
COLLECTIONS_DIR = DEFAULT_DATA_DIR / "collections"
SAMPLES_DIR = DEFAULT_DATA_DIR / "samples"

# Ensure essential directories exist
for path in (DEFAULT_DATA_DIR, WORKSPACE_DIR, CACHE_DIR, MODELS_DIR, NOTES_DIR, COLLECTIONS_DIR, SAMPLES_DIR):
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelPreset:
    id: str
    name: str
    repo_id: str
    hf_filename: str
    local_filename: str
    context_length: int
    size_label: str
    supports_vision: bool = False
    mmproj_repo_id: Optional[str] = None
    mmproj_hf_filename: Optional[str] = None
    mmproj_local_filename: Optional[str] = None

    @property
    def local_path(self) -> Path:
        return MODELS_DIR / self.local_filename

    @property
    def mmproj_local_path(self) -> Optional[Path]:
        if self.mmproj_local_filename:
            return MODELS_DIR / self.mmproj_local_filename
        return None

    @property
    def is_downloaded(self) -> bool:
        return self.local_path.exists()

    @property
    def is_vision_ready(self) -> bool:
        return bool(self.supports_vision and self.mmproj_local_path and self.mmproj_local_path.exists())


GEMMA_MODELS: Dict[str, ModelPreset] = {
    "gemma-4-e2b": ModelPreset(
        id="gemma-4-e2b",
        name="Gemma 4 E2B Multimodal (Ultra-Rápido + Visión)",
        repo_id="unsloth/gemma-4-E2B-it-GGUF",
        hf_filename="gemma-4-E2B-it-Q4_K_M.gguf",
        local_filename="gemma_4_e2b_model.gguf",
        context_length=32768,
        size_label="2.9 GB",
        supports_vision=True,
        mmproj_repo_id="unsloth/gemma-4-E2B-it-GGUF",
        mmproj_hf_filename="mmproj-F16.gguf",
        mmproj_local_filename="mmproj_gemma_4_e2b.gguf",
    ),
    "gemma-4-e4b": ModelPreset(
        id="gemma-4-e4b",
        name="Gemma 4 E4B (Equilibrado 128K)",
        repo_id="unsloth/gemma-4-E4B-it-GGUF",
        hf_filename="gemma-4-E4B-it-Q4_K_M.gguf",
        local_filename="gemma_4_e4b_model.gguf",
        context_length=32768,
        size_label="3.3 GB",
        supports_vision=False,
    ),
    "gemma-4-12b": ModelPreset(
        id="gemma-4-12b",
        name="Gemma 4 12B (Máxima Calidad Profunda)",
        repo_id="lmstudio-community/gemma-4-12B-it-GGUF",
        hf_filename="gemma-4-12B-it-Q4_K_M.gguf",
        local_filename="gemma_4_12b_model.gguf",
        context_length=32768,
        size_label="6.9 GB",
        supports_vision=True,
        mmproj_repo_id="lmstudio-community/gemma-4-12B-it-GGUF",
        mmproj_hf_filename="mmproj-gemma-4-12B-it-BF16.gguf",
        mmproj_local_filename="mmproj_gemma_4_12b.gguf",
    ),
    "gemma-2-2b": ModelPreset(
        id="gemma-2-2b",
        name="Gemma 2 2B (Ligero Clásico)",
        repo_id="bartowski/gemma-2-2b-it-GGUF",
        hf_filename="gemma-2-2b-it-Q4_K_M.gguf",
        local_filename="gemma_model.gguf",
        context_length=8192,
        size_label="1.6 GB",
        supports_vision=False,
    ),
}

DEFAULT_MODEL_ID = "gemma-4-e2b"
DEFAULT_MODEL = GEMMA_MODELS[DEFAULT_MODEL_ID]
LARGER_MODEL = GEMMA_MODELS["gemma-4-12b"]
LOCAL_GGUF_PATH = DEFAULT_MODEL.local_path
