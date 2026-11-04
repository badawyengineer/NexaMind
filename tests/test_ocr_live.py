import os

import pytest
from PIL import Image, ImageDraw, ImageFont

from genai_engineering.task2_rag import ingestion

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_OCR_TESTS") != "1",
    reason="set RUN_OCR_TESTS=1 to exercise the installed Tesseract binary",
)
def test_real_tesseract_extracts_text_from_a_scanned_pdf(tmp_path):
    image = Image.new("RGB", (1800, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (80, 150),
        "NexaMind OCR Test 73042",
        fill="black",
        font=ImageFont.load_default(size=64),
    )
    image_path = tmp_path / "scan.png"
    image.save(image_path)
    pdf_path = tmp_path / "scanned.pdf"
    document = ingestion.pymupdf.open()
    page = document.new_page(width=900, height=250)
    page.insert_image(page.rect, stream=image_path.read_bytes())
    document.save(pdf_path)
    document.close()

    pages = ingestion.load_document(pdf_path, ocr_enabled=True, ocr_language="eng")

    assert len(pages) == 1
    assert pages[0].extraction_method == "ocr"
    assert "73042" in pages[0].text
