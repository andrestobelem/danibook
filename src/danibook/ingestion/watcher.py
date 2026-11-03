from pathlib import Path
from typing import Dict, List, Optional
import json
from .parser import DocumentParser, ParsedDocument

class FolderWatcher:
    """Monitors a local folder and incrementally parses new or changed documents based on SHA-256."""

    SUPPORTED_EXTENSIONS = DocumentParser.SUPPORTED_EXTENSIONS

    def __init__(self, watch_dir: Path, state_file: Optional[Path] = None):
        self.watch_dir = Path(watch_dir).resolve()
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = state_file or (self.watch_dir / ".danibook_state.json")
        self.file_hashes: Dict[str, str] = self._load_state()

    def _load_state(self) -> Dict[str, str]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_state(self):
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.file_hashes, f, indent=2)
        except Exception:
            pass

    def scan(self) -> List[ParsedDocument]:
        """Scans the folder and returns parsed documents for newly added or modified files."""
        changed_docs: List[ParsedDocument] = []
        current_files = set()

        for file_path in self.watch_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                if file_path.name.startswith("."):
                    continue
                current_files.add(file_path.name)
                current_hash = DocumentParser.calculate_file_hash(file_path)

                if self.file_hashes.get(file_path.name) != current_hash:
                    # New or changed file
                    parsed = DocumentParser.parse(file_path)
                    changed_docs.append(parsed)
                    self.file_hashes[file_path.name] = current_hash

        # Clean up deleted files from state
        deleted_files = set(self.file_hashes.keys()) - current_files
        for del_file in deleted_files:
            del self.file_hashes[del_file]

        if changed_docs or deleted_files:
            self._save_state()

        return changed_docs
