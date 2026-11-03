from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import base64
import csv
import hashlib
import json
import mimetypes
import re
import urllib.parse
import urllib.request
import pypdf
import docx
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


@dataclass
class DocumentChunk:
    source_file: str
    content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass
class ParsedDocument:
    source_file: str
    file_path: Path
    file_hash: str
    chunks: List[DocumentChunk]
    full_text: str
    media_type: str = "text"  # "text", "image", "audio", "web", "youtube", "table"
    image_data_uri: Optional[str] = None


class DocumentParser:
    """Parses all NotebookLM supported formats (.pdf, .docx, .epub, .md, .txt, .csv, .tsv, .json, .html, images, audio, URLs, YouTube)."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".aac"}
    TABLE_EXTENSIONS = {".csv", ".tsv"}
    TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".json", ".html", ".htm"}
    DOC_EXTENSIONS = {".pdf", ".docx", ".epub"}

    SUPPORTED_EXTENSIONS = (
        IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | TABLE_EXTENSIONS | TEXT_EXTENSIONS | DOC_EXTENSIONS
    )

    @staticmethod
    def calculate_file_hash(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    @classmethod
    def parse(cls, path: Path) -> ParsedDocument:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_hash = cls.calculate_file_hash(path)
        ext = path.suffix.lower()

        if ext in (".md", ".markdown", ".txt"):
            chunks = cls._parse_markdown(path)
            media_type = "text"
            image_data_uri = None
        elif ext == ".pdf":
            chunks = cls._parse_pdf(path)
            media_type = "text"
            image_data_uri = None
        elif ext == ".docx":
            chunks = cls._parse_docx(path)
            media_type = "text"
            image_data_uri = None
        elif ext == ".epub":
            chunks = cls._parse_epub(path)
            media_type = "text"
            image_data_uri = None
        elif ext in cls.TABLE_EXTENSIONS:
            chunks = cls._parse_csv(path, delimiter="\t" if ext == ".tsv" else ",")
            media_type = "table"
            image_data_uri = None
        elif ext in (".html", ".htm"):
            chunks = cls._parse_html(path)
            media_type = "web"
            image_data_uri = None
        elif ext == ".json":
            chunks = cls._parse_json(path)
            media_type = "text"
            image_data_uri = None
        elif ext in cls.IMAGE_EXTENSIONS:
            chunks, image_data_uri = cls._parse_image(path)
            media_type = "image"
        elif ext in cls.AUDIO_EXTENSIONS:
            chunks = cls._parse_audio(path)
            media_type = "audio"
            image_data_uri = None
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. Supported formats: {sorted(cls.SUPPORTED_EXTENSIONS)}"
            )

        full_text = "\n\n".join(chunk.content for chunk in chunks)
        return ParsedDocument(
            source_file=path.name,
            file_path=path,
            file_hash=file_hash,
            chunks=chunks,
            full_text=full_text,
            media_type=media_type,
            image_data_uri=image_data_uri,
        )

    @classmethod
    def _parse_markdown(cls, path: Path) -> List[DocumentChunk]:
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

    @classmethod
    def _parse_pdf(cls, path: Path, max_pages: int = 20) -> List[DocumentChunk]:
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
                        )
                    )
            except Exception:
                continue
        if not chunks:
            chunks.append(
                DocumentChunk(
                    source_file=path.name,
                    content=f"[Documento PDF {path.name} ({total_pages} páginas)]",
                    page_number=1,
                )
            )
        return chunks

    @classmethod
    def _parse_docx(cls, path: Path) -> List[DocumentChunk]:
        doc = docx.Document(str(path))
        chunks = []
        current_section = None
        current_paras = []

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            if p.style.name.startswith("Heading"):
                if current_paras:
                    chunks.append(
                        DocumentChunk(
                            source_file=path.name,
                            content="\n".join(current_paras),
                            section_title=current_section,
                        )
                    )
                    current_paras = []
                current_section = text
            current_paras.append(text)

        if current_paras:
            chunks.append(
                DocumentChunk(
                    source_file=path.name,
                    content="\n".join(current_paras),
                    section_title=current_section,
                )
            )

        return chunks

    @classmethod
    def _parse_epub(cls, path: Path) -> List[DocumentChunk]:
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
        chunks = []
        chapter_idx = 1

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text = soup.get_text(separator="\n").strip()
                if not text:
                    continue

                header = soup.find(["h1", "h2", "h3"])
                section_title = header.get_text().strip() if header else f"Capítulo {chapter_idx}"

                # Split massive EPUB documents into ~3500-char sub-chunks for granular BM25 retrieval
                max_chunk_chars = 3500
                if len(text) <= max_chunk_chars:
                    chunks.append(
                        DocumentChunk(
                            source_file=path.name,
                            content=text,
                            page_number=chapter_idx,
                            section_title=section_title,
                        )
                    )
                    chapter_idx += 1
                else:
                    paragraphs = text.split("\n")
                    buf = []
                    buf_len = 0
                    for para in paragraphs:
                        if buf_len + len(para) > max_chunk_chars and buf:
                            chunks.append(
                                DocumentChunk(
                                    source_file=path.name,
                                    content="\n".join(buf),
                                    page_number=chapter_idx,
                                    section_title=section_title,
                                )
                            )
                            chapter_idx += 1
                            buf = []
                            buf_len = 0
                        buf.append(para)
                        buf_len += len(para) + 1
                    if buf:
                        chunks.append(
                            DocumentChunk(
                                source_file=path.name,
                                content="\n".join(buf),
                                page_number=chapter_idx,
                                section_title=section_title,
                            )
                        )
                        chapter_idx += 1

        return chunks

    @classmethod
    def _parse_csv(cls, path: Path, delimiter: str = ",") -> List[DocumentChunk]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = list(csv.reader(f, delimiter=delimiter))
        if not reader:
            return []

        headers = reader[0]
        rows = reader[1:]
        chunks = []
        batch_size = 40

        for idx in range(0, max(1, len(rows)), batch_size):
            batch = rows[idx : idx + batch_size]
            md_lines = [
                f"### Tabla de Datos: {path.name} (Filas {idx + 1} a {idx + len(batch)})",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            for r in batch:
                padded = r + [""] * max(0, len(headers) - len(r))
                md_lines.append("| " + " | ".join(padded[: len(headers)]) + " |")

            chunks.append(
                DocumentChunk(
                    source_file=path.name,
                    content="\n".join(md_lines),
                    page_number=(idx // batch_size) + 1,
                    section_title=f"Filas {idx + 1}-{idx + len(batch)}",
                )
            )
        return chunks

    @classmethod
    def _parse_html(cls, path: Path) -> List[DocumentChunk]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.get_text().strip() if soup.title else path.name
        text = soup.get_text(separator="\n").strip()
        return [
            DocumentChunk(
                source_file=path.name,
                content=f"# {title}\n\n{text}",
                section_title=title,
            )
        ]

    @classmethod
    def _parse_json(cls, path: Path) -> List[DocumentChunk]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            obj = json.loads(raw)
            formatted = json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            formatted = raw
        return [
            DocumentChunk(
                source_file=path.name,
                content=f"```json\n{formatted}\n```",
                section_title="Datos JSON",
            )
        ]

    @classmethod
    def _parse_image(cls, path: Path) -> Tuple[List[DocumentChunk], str]:
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "image/png"
        raw_bytes = path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
        size_kb = len(raw_bytes) / 1024.0
        desc = (
            f"[Fuente de Imagen Multimodal: {path.name} ({mime}, {size_kb:.1f} KB)]\n"
            f"Esta imagen está activa en el cuaderno y puede ser inspeccionada visualmente por el modelo multimodal Gemma 4."
        )
        return [
            DocumentChunk(
                source_file=path.name,
                content=desc,
                section_title="Imagen Multimodal",
            )
        ], data_uri

    @classmethod
    def _parse_audio(cls, path: Path) -> List[DocumentChunk]:
        size_kb = path.stat().st_size / 1024.0
        desc = (
            f"# Fuente de Audio: {path.name}\n\n"
            f"- **Formato**: `{path.suffix.lower()}`\n"
            f"- **Tamaño**: `{size_kb:.1f} KB`\n"
            f"- **Estado**: Archivo de audio registrado en el cuaderno local."
        )
        return [
            DocumentChunk(
                source_file=path.name,
                content=desc,
                section_title="Fuente de Audio",
            )
        ]

    @classmethod
    def ingest_url_to_file(cls, url: str, workspace_dir: Path) -> Path:
        """Fetches a Web URL or YouTube video page/transcript and saves it as a structured Markdown source."""
        parsed_url = urllib.parse.urlparse(url.strip())
        if not parsed_url.scheme:
            url = "https://" + url.strip()
            parsed_url = urllib.parse.urlparse(url)

        is_youtube = any(h in parsed_url.netloc.lower() for h in ("youtube.com", "youtu.be"))
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            )
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            html_bytes = resp.read()
            html_text = html_bytes.decode("utf-8", errors="replace")

        soup = BeautifulSoup(html_text, "html.parser")
        page_title = soup.title.get_text().strip() if soup.title else parsed_url.netloc

        if is_youtube:
            # Extract video description & metadata from YouTube HTML
            meta_desc = soup.find("meta", attrs={"name": "description"})
            desc_text = meta_desc["content"] if meta_desc and meta_desc.get("content") else ""
            # Also try extracting caption tracks if embedded in ytInitialPlayerResponse
            captions_text = ""
            m = re.search(r'"captionTracks":(\[.*?\])', html_text)
            if m:
                try:
                    tracks = json.loads(m.group(1))
                    if tracks and "baseUrl" in tracks[0]:
                        cap_req = urllib.request.Request(tracks[0]["baseUrl"], headers=headers)
                        with urllib.request.urlopen(cap_req, timeout=10) as cap_resp:
                            cap_xml = cap_resp.read().decode("utf-8", errors="replace")
                            cap_soup = BeautifulSoup(cap_xml, "html.parser")
                            captions_text = "\n".join(t.get_text() for t in cap_soup.find_all("text"))
                except Exception:
                    pass

            safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", page_title)[:45].strip("_") or "youtube_video"
            dest = workspace_dir / f"YouTube_{safe_slug}.md"
            md = [
                f"# Video de YouTube: {page_title}",
                f"- **URL Original**: {url}",
                "",
                "## Descripción del Video",
                desc_text or "Sin descripción pública.",
            ]
            if captions_text:
                md.extend(["", "## Transcripción / Subtítulos", captions_text])
            dest.write_text("\n".join(md), encoding="utf-8")
            return dest

        # Generic Web Article extraction
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            tag.decompose()

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["h1", "h2", "h3", "p", "li", "blockquote"])]
        clean_paragraphs = [p for p in paragraphs if len(p) > 25]
        body_text = "\n\n".join(clean_paragraphs) if clean_paragraphs else soup.get_text(separator="\n").strip()

        safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", page_title)[:45].strip("_") or "web_source"
        dest = workspace_dir / f"Web_{safe_slug}.md"
        md_content = f"# Fuente Web: {page_title}\n\n- **URL**: {url}\n\n---\n\n{body_text}"
        dest.write_text(md_content, encoding="utf-8")
        return dest
