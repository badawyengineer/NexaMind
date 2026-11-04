import io

from PIL import Image

from genai_engineering.task2_rag import ingestion


def test_pdf_pages_keep_one_based_numbers_and_method(tmp_path):
    path = tmp_path / "manual.pdf"
    document = ingestion.pymupdf.open()
    for text in ("First page text", "Second page text"):
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()

    pages = ingestion.load_document(path)

    assert [page.page for page in pages] == [1, 2]
    assert all(page.source == "manual.pdf" for page in pages)
    assert all(page.extraction_method == "text" for page in pages)


def test_scanned_pdf_uses_mocked_ocr_and_records_method(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    document = ingestion.pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()
    monkeypatch.setattr(ingestion, "_ocr_image", lambda *args: "recognized words")

    pages = ingestion.load_document(path, ocr_enabled=True)

    assert len(pages) == 1
    assert pages[0].text == "recognized words"
    assert pages[0].extraction_method == "ocr"


def test_scanned_pdf_without_ocr_fails_clearly(tmp_path):
    image_bytes = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(image_bytes, format="PNG")
    path = tmp_path / "scan.pdf"
    document = ingestion.pymupdf.open()
    page = document.new_page()
    page.insert_image(page.rect, stream=image_bytes.getvalue())
    document.save(path)
    document.close()

    try:
        ingestion.load_document(path, ocr_enabled=False)
    except ValueError as exc:
        assert "enable OCR" in str(exc)
    else:
        raise AssertionError("image-based PDF was silently accepted")


def test_image_ocr_uses_mock_and_enforces_pixel_limit(tmp_path, monkeypatch):
    path = tmp_path / "scan.png"
    Image.new("RGB", (20, 20), "white").save(path)
    monkeypatch.setattr(ingestion, "_ocr_image", lambda *args: "image text")

    page = ingestion.load_document(path, ocr_enabled=True)[0]

    assert page.extraction_method == "ocr"
    assert page.text == "image text"
    try:
        ingestion.load_document(path, max_image_pixels=100)
    except ValueError as exc:
        assert "pixel limit" in str(exc)
    else:
        raise AssertionError("oversized image was accepted")


def test_rejects_mismatched_mime_and_page_limit(tmp_path):
    wrong_pdf = tmp_path / "wrong.pdf"
    wrong_pdf.write_text("not a PDF")
    try:
        ingestion.load_document(wrong_pdf)
    except ValueError as exc:
        assert "match PDF" in str(exc)
    else:
        raise AssertionError("mismatched MIME was accepted")

    path = tmp_path / "many.pdf"
    doc = ingestion.pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(path)
    doc.close()
    try:
        ingestion.load_document(path, max_pages=1)
    except ValueError as exc:
        assert "page limit" in str(exc)
    else:
        raise AssertionError("over-limit PDF was accepted")
