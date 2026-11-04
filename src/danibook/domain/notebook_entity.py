from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
import datetime
import json
import uuid


@dataclass
class Notebook:
    """Notebook Aggregate Root representing an isolated research session, workspace, and notes."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = "Cuaderno Sin Título"
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    base_dir: Optional[Path] = None

    @property
    def root_dir(self) -> Path:
        if self.base_dir:
            return self.base_dir / "notebooks" / self.id
        return Path("data/notebooks") / self.id

    @property
    def workspace_dir(self) -> Path:
        return self.root_dir / "workspace"

    @property
    def notes_dir(self) -> Path:
        return self.root_dir / "notes"

    @property
    def meta_file(self) -> Path:
        return self.root_dir / "notebook.json"

    def ensure_directories(self) -> None:
        """Creates the workspace and notes directories for this notebook on disk."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.save_metadata()

    def save_metadata(self) -> None:
        """Persists the Notebook metadata to disk."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
        }
        self.meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
        }

    @classmethod
    def load_from_dir(cls, root_dir: Path, base_dir: Optional[Path] = None) -> "Notebook":
        meta_file = root_dir / "notebook.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                return cls(
                    id=data.get("id", root_dir.name),
                    title=data.get("title", "Cuaderno"),
                    created_at=data.get("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
                    base_dir=base_dir,
                )
            except Exception:
                pass
        return cls(id=root_dir.name, title="Cuaderno", base_dir=base_dir)
