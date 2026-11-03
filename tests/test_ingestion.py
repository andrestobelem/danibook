import pytest
from pathlib import Path
import docx
from danibook.ingestion.parser import DocumentParser, DocumentChunk

def test_parse_markdown(tmp_path):
    md_file = tmp_path / "capitulo_1.md"
    md_file.write_text("# Capítulo 1\nEl inicio del viaje.\n# Capítulo 2\nEl desenlace.", encoding="utf-8")

    doc = DocumentParser.parse(md_file)
    assert doc.source_file == "capitulo_1.md"
    assert len(doc.chunks) == 2
    assert "El inicio del viaje." in doc.chunks[0].content
    assert doc.chunks[0].section_title == "Capítulo 1"
    assert doc.chunks[1].section_title == "Capítulo 2"

def test_parse_docx(tmp_path):
    docx_file = tmp_path / "borrador.docx"
    doc = docx.Document()
    doc.add_heading("Prólogo", level=1)
    doc.add_paragraph("Este es el texto del prólogo.")
    doc.save(str(docx_file))

    parsed = DocumentParser.parse(docx_file)
    assert parsed.source_file == "borrador.docx"
    assert len(parsed.chunks) == 1
    assert "Este es el texto del prólogo." in parsed.chunks[0].content
    assert parsed.chunks[0].section_title == "Prólogo"


def test_parse_csv_and_image(tmp_path):
    csv_file = tmp_path / "datos.csv"
    csv_file.write_text("personaje,obra\nDon Quijote,El Quijote\nSancho Panza,El Quijote\n", encoding="utf-8")
    parsed_csv = DocumentParser.parse(csv_file)
    assert parsed_csv.media_type == "table"
    assert "Don Quijote" in parsed_csv.full_text

    # 1x1 valid PNG bytes
    import base64
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    img_file = tmp_path / "diagrama.png"
    img_file.write_bytes(png_bytes)
    parsed_img = DocumentParser.parse(img_file)
    assert parsed_img.media_type == "image"
    assert parsed_img.image_data_uri.startswith("data:image/png;base64,")
