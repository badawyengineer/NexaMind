from types import SimpleNamespace

from genai_engineering.task2_rag import ingestion


def test_pdf_pages_keep_one_based_page_numbers(tmp_path, monkeypatch):
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(
        ingestion,
        "PdfReader",
        lambda _: SimpleNamespace(
            pages=[
                SimpleNamespace(extract_text=lambda: "First page text"),
                SimpleNamespace(extract_text=lambda: "Second page text"),
            ]
        ),
    )

    pages = ingestion.load_document(path)

    assert [page.page for page in pages] == [1, 2]
    assert all(page.source == "manual.pdf" for page in pages)
