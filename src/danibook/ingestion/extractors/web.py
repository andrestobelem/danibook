from pathlib import Path
import re
import urllib.request


class WebIngestor:
    """Fetches web URLs or YouTube video transcripts and saves them as Markdown sources."""

    @staticmethod
    def ingest_url_to_file(url: str, workspace_dir: Path) -> Path:
        clean_url = url.strip()
        is_youtube = "youtube.com" in clean_url or "youtu.be" in clean_url

        if is_youtube:
            video_id = None
            if "v=" in clean_url:
                video_id = clean_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in clean_url:
                video_id = clean_url.split("youtu.be/")[1].split("?")[0]

            title = f"YouTube_{video_id or 'Video'}"
            content = f"# Transcripción de Video de YouTube\n\nURL: {clean_url}\n\n"
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                if video_id:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["es", "en"])
                    body = "\n".join(f"[{int(entry['start'])}s] {entry['text']}" for entry in transcript_list)
                    content += body
                else:
                    content += "No se pudo extraer el identificador del video."
            except Exception as e:
                content += f"No se pudo descargar la transcripción automática: {e}\n(El audio puede procesarse directamente con Gemma 4 Multimodal)."
        else:
            title_slug = re.sub(r"[^\w]", "_", clean_url.replace("https://", "").replace("http://", ""))[:40]
            title = f"Web_{title_slug}"
            try:
                from bs4 import BeautifulSoup
                req = urllib.request.Request(clean_url, headers={"User-Agent": "DaniBookLM/0.2 (Research Bot)"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html_bytes = resp.read()
                soup = BeautifulSoup(html_bytes, "html.parser")
                h1 = soup.find(["h1", "title"])
                page_title = h1.get_text().strip() if h1 else title
                for tag in soup(["script", "style", "nav", "footer", "aside"]):
                    tag.decompose()
                body = soup.get_text(separator="\n").strip()
                clean_body = re.sub(r"\n{3,}", "\n\n", body)
                content = f"# {page_title}\n\nURL Original: {clean_url}\n\n{clean_body}"
            except Exception as e:
                content = f"# Página Web: {clean_url}\n\nError al descargar contenido en vivo: {e}"

        dest = (workspace_dir / f"{title}.md").resolve()
        dest.write_text(content, encoding="utf-8")
        return dest
