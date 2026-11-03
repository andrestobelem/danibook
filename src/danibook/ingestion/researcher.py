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
                "title": f"Búsqueda sobre: {topic}",
                "snippet": f"No se pudo completar la búsqueda en vivo ({e}). Puedes redactar o pegar apuntes manualmente.",
                "url": "local://offline",
            })
        return results

    def compile_research_dossier(self, topic: str) -> str:
        """Synthesizes search findings into a structured, cited research markdown report."""
        findings = self.search_web(topic)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"# INFORME DE DEEP RESEARCH: {topic.upper()}",
            f"*Fecha de investigación: {timestamp}*",
            "\n## Síntesis de Hallazgos y Fuentes Consultadas\n"
        ]

        for i, item in enumerate(findings, start=1):
            lines.append(f"### {i}. {item['title']}")
            lines.append(f"**Fuente:** [{item['url']}]({item['url']})\n")
            lines.append(f"> {item['snippet']}\n")

        lines.append("## Conclusiones para la Redacción")
        lines.append(
            "- Este documento fue generado mediante Deep Research y se incorporó a las fuentes del libro.\n"
            "- Puedes contrastarlo con tus borradores activos para verificar coherencia de datos, fechas y ambientación."
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
