"""Bounded text and OCR ingestion for PDFs, text files, and images."""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class Page:
    text: str
    source: str
    source_id: str
    page: int
    extraction_method: str = "text"


def _ocr_image(image: Image.Image, language: str, tesseract_cmd: str | None) -> str:
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise RuntimeError("OCR support requires pytesseract") from exc
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        pytesseract.get_tesseract_version()
        return pytesseract.image_to_string(image, lang=language).strip()
    except Exception as exc:
        raise RuntimeError(
            "OCR failed. Install Tesseract and the requested language data, "
            f"then verify OCR_LANGUAGE={language!r}."
        ) from exc


def load_document(
    path: str | Path,
    *,
    max_bytes: int = 20_000_000,
    max_pages: int = 100,
    max_image_pixels: int = 40_000_000,
    ocr_enabled: bool = False,
    ocr_language: str = "eng",
    ocr_fallback_threshold: int = 20,
    ocr_dpi: int = 200,
    tesseract_cmd: str | None = None,
) -> list[Page]:
    file = Path(path)
    suffix = file.suffix.lower()
    if not file.is_file() or suffix not in {".pdf", ".txt", ".png", ".jpg", ".jpeg"}:
        raise ValueError("Only existing PDF, TXT, PNG, and JPEG files are supported")
    size = file.stat().st_size
    if size > max_bytes:
        raise ValueError("Document exceeds the configured size limit")
    raw = file.read_bytes()
    if suffix == ".pdf" and not raw.startswith(b"%PDF-"):
        raise ValueError("File content does not match PDF format")
    if suffix == ".png" and not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("File content does not match PNG format")
    if suffix in {".jpg", ".jpeg"} and not raw.startswith(b"\xff\xd8\xff"):
        raise ValueError("File content does not match JPEG format")
    digest = hashlib.sha256(raw).hexdigest()

    if suffix == ".txt":
        if b"\x00" in raw:
            raise ValueError("File content does not match plain-text format")
        text = raw.decode("utf-8", errors="replace").strip()
        pages = [Page(text, file.name, digest, 1, "text")]
    elif suffix in {".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > max_image_pixels:
                    raise ValueError("Image exceeds the configured pixel limit")
                image.load()
                if not ocr_enabled:
                    raise ValueError(
                        "Image OCR is disabled; enable OCR to ingest images"
                    )
                text = _ocr_image(image.convert("RGB"), ocr_language, tesseract_cmd)
        except UnidentifiedImageError as exc:
            raise ValueError("Image content is invalid") from exc
        pages = [Page(text, file.name, digest, 1, "ocr")]
    else:
        pages = []
        try:
            with pymupdf.open(stream=raw, filetype="pdf") as document:
                if document.page_count > max_pages:
                    raise ValueError("PDF exceeds the configured page limit")
                for index, pdf_page in enumerate(document, 1):
                    text = pdf_page.get_text("text").strip()
                    method = "text"
                    has_images = bool(pdf_page.get_images(full=True))
                    if (
                        not ocr_enabled
                        and len(text) < ocr_fallback_threshold
                        and has_images
                    ):
                        raise ValueError(
                            f"PDF page {index} contains image-based content; "
                            "enable OCR to ingest it"
                        )
                    if ocr_enabled and len(text) < ocr_fallback_threshold:
                        estimated_width = pdf_page.rect.width * ocr_dpi / 72
                        estimated_height = pdf_page.rect.height * ocr_dpi / 72
                        if estimated_width * estimated_height > max_image_pixels:
                            raise ValueError(
                                f"Rendered page {index} exceeds the configured "
                                "pixel limit"
                            )
                        pixmap = pdf_page.get_pixmap(dpi=ocr_dpi, alpha=False)
                        if pixmap.width * pixmap.height > max_image_pixels:
                            raise ValueError(
                                f"Rendered page {index} exceeds the configured "
                                "pixel limit"
                            )
                        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                        text = _ocr_image(image, ocr_language, tesseract_cmd)
                        method = "ocr"
                    pages.append(Page(text, file.name, digest, index, method))
        except pymupdf.FileDataError as exc:
            raise ValueError("PDF content is invalid or encrypted") from exc
    pages = [page for page in pages if page.text]
    if not pages:
        raise ValueError("Document contains no extractable text")
    return pages
