from typing import List, Dict
from pathlib import Path
from datetime import datetime
from duckduckgo_search import DDGS
from ..config import WORKSPACE_DIR
from ..ingestion.parser import DocumentParser, ParsedDocument

class DeepResearcher:
    """Performs multi-query autonomous web research and synthesizes cited research dossiers."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search_web(self, topic: str) -> List[Dict[str, str]]:
        """Queries the web for background information, snippets, and source URLs."""
        results = []
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(topic, max_results=self.max_results))
                for item in raw_results:
                    results.append({
                        "title": item.get("title", "Sin título"),
                        "snippet": item.get("body", ""),
                        "url": item.get("href", ""),
                    })
        except Exception as e:
            # Fallback for offline or throttled connection
            results.append({
                "title": f"Research on: {topic}",
                "snippet": f"Web search could not be completed at this moment ({e}). You can edit or paste research notes manually.",
                "url": "local://offline",
            })
        return results

    def compile_research_dossier(self, topic: str) -> str:
        """Synthesizes search findings into a structured, cited research markdown report."""
        findings = self.search_web(topic)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"# INFORME DE DEEP RESEARCH: {topic.upper()}",
            f"*Research timestamp: {timestamp}*",
            "\n## Síntesis de Hallazgos y Fuentes Consultadas\n"
        ]

        for i, item in enumerate(findings, start=1):
            lines.append(f"### {i}. {item['title']}")
            lines.append(f"**Source:** [{item['url']}]({item['url']})\n")
            lines.append(f"> {item['snippet']}\n")

        lines.append("## Research Synthesis & Editorial Notes")
        lines.append(
            "- This document was compiled autonomously via Deep Research and added to the notebook sources.\n"
            "- You can query it in the chat or contrast it against active sources to evaluate facts, dates, and arguments."
        )

        return "\n".join(lines)

    def import_to_workspace(self, topic: str, dossier_markdown: str) -> Path:
        """Saves the synthesized research dossier directly into the book's workspace folder."""
        clean_name = "".join(c for c in topic if c.isalnum() or c in ("-", "_", " ")).strip().replace(" ", "_")
        filename = f"research_{clean_name[:40].lower()}.md"
        file_path = WORKSPACE_DIR / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(dossier_markdown)
        return file_path
