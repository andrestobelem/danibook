from typing import List, Dict, Tuple, Optional
import logging
import re
import sqlite3
from pathlib import Path
from ..ingestion.parser import ParsedDocument, DocumentChunk
from ..config import CACHE_DIR

logger = logging.getLogger(__name__)


class ArchiveMemory:
    """Tier 2: Archivo / Biblioteca de Referencia con SQLite FTS5 BM25 Nativo.

    Almacena e indexa fragmentos documentales en SQLite FTS5 para búsqueda
    léxica-semántica instantánea y persistente en disco sin recalcular hashes en cada arranque.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (CACHE_DIR / "archive_fts5.sqlite")
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_schema()
        self.indexed_files: Dict[str, str] = self._load_indexed_files()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS indexed_files (
                    source_file TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    source_file,
                    content,
                    page_number UNINDEXED,
                    section_title UNINDEXED,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
            """)

    def _load_indexed_files(self) -> Dict[str, str]:
        cur = self.conn.execute("SELECT source_file, file_hash FROM indexed_files")
        return {row[0]: row[1] for row in cur.fetchall()}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def add_document(self, doc: ParsedDocument):
        """Incrementally indexes a document into SQLite FTS5. Skips if hash matches."""
        if self.indexed_files.get(doc.source_file) == doc.file_hash:
            return

        self.remove_document(doc.source_file)

        rows = [
            (
                c.source_file,
                c.content,
                c.page_number,
                c.section_title,
            )
            for c in doc.chunks
            if c.content and c.content.strip()
        ]
        with self.conn:
            if rows:
                self.conn.executemany(
                    "INSERT INTO chunks_fts (source_file, content, page_number, section_title) VALUES (?, ?, ?, ?)",
                    rows,
                )
            self.conn.execute(
                "INSERT OR REPLACE INTO indexed_files (source_file, file_hash) VALUES (?, ?)",
                (doc.source_file, doc.file_hash),
            )
        self.indexed_files[doc.source_file] = doc.file_hash

    def remove_document(self, source_file: str):
        with self.conn:
            self.conn.execute("DELETE FROM chunks_fts WHERE source_file = ?", (source_file,))
            self.conn.execute("DELETE FROM indexed_files WHERE source_file = ?", (source_file,))
        if source_file in self.indexed_files:
            del self.indexed_files[source_file]

    def load_cached_document(self, path: Path, file_hash: str) -> Optional[ParsedDocument]:
        """Reconstructs a ParsedDocument from SQLite FTS5 cache in <1ms without re-parsing binary PDFs."""
        if self.indexed_files.get(path.name) != file_hash:
            return None
        cur = self.conn.execute(
            "SELECT content, page_number, section_title FROM chunks_fts WHERE source_file = ?",
            (path.name,),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        chunks = [
            DocumentChunk(
                source_file=path.name,
                content=r[0],
                page_number=r[1],
                section_title=r[2],
            )
            for r in rows
        ]
        full_text = "\n\n".join(c.content for c in chunks)
        return ParsedDocument(
            source_file=path.name,
            file_path=path,
            file_hash=file_hash,
            chunks=chunks,
            full_text=full_text,
            media_type="text",
            image_data_uri=None,
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Performs native SQLite FTS5 BM25 retrieval across all reference chunks."""
        query_tokens = [t for t in self._tokenize(query) if len(t) >= 2]
        if not query_tokens:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in query_tokens[:12])
        try:
            cur = self.conn.execute(
                """
                SELECT source_file, content, page_number, section_title, -bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (fts_query, top_k),
            )
            results: List[Tuple[DocumentChunk, float]] = []
            for row in cur.fetchall():
                raw_content = "".join(ch for ch in (row[1] or "") if ch == "\n" or ch == "\t" or ord(ch) >= 32)
                chunk = DocumentChunk(
                    source_file=row[0],
                    content=raw_content,
                    page_number=row[2],
                    section_title=row[3],
                )
                results.append((chunk, float(row[4] or 1.0)))
            return results
        except Exception:
            logger.warning("Archive FTS5 search failed for query=%r", query, exc_info=True)
            return []

    def format_search_context(self, query: str, top_k: int = 4, max_chunk_chars: int = 2500) -> str:
        """Formats the retrieved fragments with clean citations for LLM injection."""
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        parts = ["# FRAGMENTOS RELEVANTES DE REFERENCIA (Biblioteca SQLite FTS5)"]
        for chunk, _ in results:
            page_info = f", Pág. {chunk.page_number}" if chunk.page_number else ""
            section_info = f", Sección: '{chunk.section_title}'" if chunk.section_title else ""
            snippet = chunk.content.strip()[:max_chunk_chars]
            parts.append(
                f"\n--- [Cita: {chunk.source_file}{page_info}{section_info}] ---\n"
                f"{snippet}"
            )
        return "\n".join(parts)
