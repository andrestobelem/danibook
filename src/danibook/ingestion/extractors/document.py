from pathlib import Path
from typing import List
import re
from .base import BaseExtractor, DocumentChunk


class DocumentExtractor(BaseExtractor):
    """Extracts chunks from rich documents: PDF, DOCX, EPUB, and HTML."""

    def extract_chunks(self, path: Path) -> List[DocumentChunk]:
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._extract_pdf(path)
        elif ext == ".docx":
            return self._extract_docx(path)
        elif ext == ".epub":
            return self._extract_epub(path)
        elif ext in (".html", ".htm"):
            return self._extract_html(path)
        return []

    def _extract_pdf(self, path: Path, max_pages: int = 20) -> List[DocumentChunk]:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path), strict=False)
            chunks = []
            total_pages = len(reader.pages)
            pages_to_read = min(total_pages, max_pages)
            for idx in range(pages_to_read):
                try:
                    page = reader.pages[idx]
                    text = page.extract_text() or ""
                    text = text.strip()
                    if text:
                        chunks.append(
                            DocumentChunk(
                                source_file=path.name,
                                content=text,
                                page_number=idx + 1,
                                section_title=f"Página {idx + 1}",
                            )
                        )
                except Exception:
                    continue
            return chunks
        except Exception:
            return []

    def _extract_docx(self, path: Path) -> List[DocumentChunk]:
        try:
            import docx
            doc = docx.Document(str(path))
            chunks = []
            current_section = "Inicio"
            current_text = []

            for p in doc.paragraphs:
                text = p.text.strip()
                if not text:
                    continue
                if p.style and p.style.name and p.style.name.startswith("Heading"):
                    if current_text:
                        chunks.append(
                            DocumentChunk(
                                source_file=path.name,
                                content="\n".join(current_text),
                                section_title=current_section,
                            )
                        )
                        current_text = []
                    current_section = text
                else:
                    current_text.append(text)

            if current_text:
                chunks.append(
                    DocumentChunk(
                        source_file=path.name,
                        content="\n".join(current_text),
                        section_title=current_section,
                    )
                )
            return chunks
        except Exception:
            return []

    def _extract_epub(self, path: Path) -> List[DocumentChunk]:
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            book = epub.read_epub(str(path))
            chunks = []
            chapter_count = 0

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    text = soup.get_text(separator="\n").strip()
                    if len(text) > 40:
                        chapter_count += 1
                        title_tag = soup.find(["h1", "h2", "h3"])
                        title = title_tag.get_text().strip() if title_tag else f"Capítulo {chapter_count}"
                        chunks.append(
                            DocumentChunk(
                                source_file=path.name,
                                content=text,
                                section_title=title,
                            )
                        )
            return chunks
        except Exception:
            return []

    def _extract_html(self, path: Path) -> List[DocumentChunk]:
        try:
            from bs4 import BeautifulSoup
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            chunks = []
            headings = soup.find_all(["h1", "h2"])
            if headings:
                for h in headings:
                    section_title = h.get_text().strip()
                    next_nodes = []
                    for sibling in h.next_siblings:
                        if getattr(sibling, "name", None) in ["h1", "h2"]:
                            break
                        text = getattr(sibling, "text", str(sibling)).strip()
                        if text:
                            next_nodes.append(text)
                    section_text = "\n".join(next_nodes).strip()
                    if section_text:
                        chunks.append(
                            DocumentChunk(
                                source_file=path.name,
                                content=f"{section_title}\n{section_text}",
                                section_title=section_title,
                            )
                        )
            if not chunks:
                raw_text = soup.get_text(separator="\n").strip()
                if raw_text:
                    chunks.append(DocumentChunk(source_file=path.name, content=raw_text))
            return chunks
        except Exception:
            return []
