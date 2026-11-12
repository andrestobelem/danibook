from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
import re
import uuid
from ..config import NOTES_DIR

logger = logging.getLogger(__name__)


class StudioStore:
    """Manages persistent Studio Notes on disk (create, list, update, delete).
    
    Separates the unique immutable identifier (`id`) from human-facing display `title`.
    Backward compatible with legacy un-frontmattered markdown files.
    """

    def __init__(self, notes_dir: Path = NOTES_DIR):
        self.notes_dir = Path(notes_dir).resolve()
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def set_notes_dir(self, new_dir: Path):
        """Switches the active notes storage directory (e.g. per notebook)."""
        self.notes_dir = Path(new_dir).resolve()
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def clear_all_notes(self):
        """Clears all notes in the active directory."""
        for path in self.notes_dir.glob("*.md"):
            try:
                path.unlink()
            except Exception:
                pass

    @staticmethod
    def _parse_note_file(path: Path) -> Dict[str, str]:
        raw = path.read_text(encoding="utf-8")
        note_id = path.stem
        title = path.stem.replace("_", " ")
        content = raw

        # Check for YAML/JSON frontmatter
        if raw.startswith("<!-- NOTE_META:"):
            end_idx = raw.find("-->")
            if end_idx != -1:
                meta_str = raw[len("<!-- NOTE_META:"):end_idx].strip()
                try:
                    meta = json.loads(meta_str)
                    note_id = meta.get("id", note_id)
                    title = meta.get("title", title)
                    content = raw[end_idx + 3:].lstrip("\r\n")
                except Exception:
                    pass

        return {
            "id": note_id,
            "title": title,
            "content": content,
        }

    def list_notes(self) -> List[Dict[str, str]]:
        """Returns all Studio Notes ordered from newest to oldest."""
        notes: List[Dict[str, str]] = []
        for path in sorted(self.notes_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                notes.append(self._parse_note_file(path))
            except Exception:
                logger.warning("Failed to parse Studio Note file '%s'", path.name, exc_info=True)
                continue
        return notes

    def save_note(
        self,
        title: str,
        content: str,
        note_id: Optional[str] = None,
        original_title: Optional[str] = None
    ) -> Path:
        """Persists a Note into the Studio directory.
        
        Uses `note_id` as the filename key (e.g. `note_uuid.md` or legacy `{clean_title}.md`).
        `title` can contain any spaces, accents or characters without affecting filename stability.
        """
        display_title = title.strip() or "Nota sin título"

        # Determine target file
        if note_id:
            clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", note_id)
            target_path = (self.notes_dir / f"{clean_id}.md").resolve()
        elif original_title:
            # Check if original_title matches an existing file stem (legacy support)
            legacy_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", original_title)
            legacy_path = (self.notes_dir / f"{legacy_stem}.md").resolve()
            if legacy_path.exists():
                target_path = legacy_path
                clean_id = legacy_stem
            else:
                clean_id = f"note_{uuid.uuid4().hex[:12]}"
                target_path = (self.notes_dir / f"{clean_id}.md").resolve()
        else:
            clean_id = f"note_{uuid.uuid4().hex[:12]}"
            target_path = (self.notes_dir / f"{clean_id}.md").resolve()

        if target_path.parent != self.notes_dir:
            raise ValueError("Invalid note path")

        meta_header = f"<!-- NOTE_META: {json.dumps({'id': clean_id, 'title': display_title}, ensure_ascii=False)} -->\n"
        full_text = meta_header + content

        target_path.write_text(full_text, encoding="utf-8")
        return target_path

    def get_note(self, note_id_or_title: str) -> Optional[Dict[str, str]]:
        """Finds a single note by id or title."""
        target_name = Path(note_id_or_title).stem
        direct_path = (self.notes_dir / f"{target_name}.md").resolve()
        if direct_path.parent == self.notes_dir and direct_path.exists():
            return self._parse_note_file(direct_path)

        for path in self.notes_dir.glob("*.md"):
            try:
                parsed = self._parse_note_file(path)
                if parsed["id"] == note_id_or_title or parsed["title"] == note_id_or_title:
                    return parsed
            except Exception:
                continue
        return None

    def delete_note(self, note_id_or_title: str) -> bool:
        """Deletes a Studio Note by note_id (stem) or legacy title."""
        target_name = Path(note_id_or_title).stem
        direct_path = (self.notes_dir / f"{target_name}.md").resolve()
        if direct_path.parent == self.notes_dir and direct_path.exists():
            direct_path.unlink()
            return True

        # Check by title inside frontmatter
        for path in self.notes_dir.glob("*.md"):
            try:
                parsed = self._parse_note_file(path)
                if parsed["id"] == note_id_or_title or parsed["title"] == note_id_or_title:
                    path.unlink()
                    return True
            except Exception:
                continue

        return False
