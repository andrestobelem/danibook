from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import datetime
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
    def workspace_dir(self) -> Path:
        if self.base_dir:
            return self.base_dir / "notebooks" / self.id / "workspace"
        return Path("data/notebooks") / self.id / "workspace"

    @property
    def notes_dir(self) -> Path:
        if self.base_dir:
            return self.base_dir / "notebooks" / self.id / "notes"
        return Path("data/notebooks") / self.id / "notes"

    def ensure_directories(self) -> None:
        """Creates the workspace and notes directories for this notebook on disk."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
