from pathlib import Path
from ebooklib import epub
from danibook.ingestion.parser import DocumentParser

def test_parse_epub(tmp_path):
    epub_path = tmp_path / "libro_test.epub"
    book = epub.EpubBook()
    book.set_identifier("test12345")
    book.set_title("Libro de Prueba")
    book.set_language("es")

    c1 = epub.EpubHtml(title="Capítulo Uno", file_name="chap_01.xhtml", lang="es")
    c1.content = "<h1>Capítulo Uno</h1><p>En un lugar de la Mancha de cuyo nombre no quiero acordarme.</p>"
    book.add_item(c1)
    book.toc = (c1,)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1]
    epub.write_epub(str(epub_path), book)

    parsed = DocumentParser.parse(epub_path)
    assert parsed.source_file == "libro_test.epub"
    assert len(parsed.chunks) >= 1
    assert "En un lugar de la Mancha" in parsed.chunks[0].content
    assert parsed.chunks[0].section_title == "Capítulo Uno"
