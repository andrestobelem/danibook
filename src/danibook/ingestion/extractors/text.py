from pathlib import Path
from typing import List
import csv
import json
from .base import BaseExtractor, DocumentChunk


class TextExtractor(BaseExtractor):
    """Extracts chunks from Markdown, plain text, CSV/TSV tables, and JSON."""

    def extract_chunks(self, path: Path) -> List[DocumentChunk]:
        ext = path.suffix.lower()
        if ext in (".csv", ".tsv"):
            return self._extract_csv(path, delimiter="\t" if ext == ".tsv" else ",")
        elif ext == ".json":
            return self._extract_json(path)
        else:
            return self._extract_markdown(path)

    def _extract_markdown(self, path: Path) -> List[DocumentChunk]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        if not content:
            return []

        sections = content.split("\n# ")
        chunks = []
        for i, section in enumerate(sections):
            text = section.strip()
            if not text:
                continue
            if i > 0:
                text = f"# {text}"
            first_line = text.split("\n", 1)[0].replace("#", "").strip()
            chunks.append(
                DocumentChunk(
                    source_file=path.name,
                    content=text,
                    section_title=first_line if first_line else None,
                )
            )
        return chunks if chunks else [DocumentChunk(source_file=path.name, content=content)]

    def _extract_csv(self, path: Path, delimiter: str = ",") -> List[DocumentChunk]:
        chunks = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)

        if not rows:
            return []

        header = rows[0]
        header_str = " | ".join(header)
        batch_size = 20
        total_rows = len(rows) - 1

        for batch_idx, start_idx in enumerate(range(1, len(rows), batch_size)):
            batch = rows[start_idx : start_idx + batch_size]
            formatted_rows = [f"Fila {start_idx + r_idx}: " + " | ".join(r) for r_idx, r in enumerate(batch)]
            content = f"Encabezados: {header_str}\n" + "\n".join(formatted_rows)
            chunks.append(
                DocumentChunk(
                    source_file=path.name,
                    content=content,
                    section_title=f"Lote {batch_idx + 1} (Filas {start_idx}-{min(start_idx + batch_size - 1, total_rows)})",
                )
            )
        return chunks

    def _extract_json(self, path: Path) -> List[DocumentChunk]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
            except Exception:
                f.seek(0)
                data = f.read()

        if isinstance(data, list):
            chunks = []
            batch_size = 10
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                text = json.dumps(batch, indent=2, ensure_ascii=False)
                chunks.append(
                    DocumentChunk(
                        source_file=path.name,
                        content=text,
                        section_title=f"Objetos {i+1} a {min(i+batch_size, len(data))}",
                    )
                )
            return chunks if chunks else [DocumentChunk(source_file=path.name, content=json.dumps(data, indent=2, ensure_ascii=False))]
        else:
            text = json.dumps(data, indent=2, ensure_ascii=False)
            return [DocumentChunk(source_file=path.name, content=text, section_title="Estructura JSON")]
