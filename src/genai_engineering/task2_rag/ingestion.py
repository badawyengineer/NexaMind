import hashlib
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    text: str
    source: str
    source_id: str
    page: int


def load_document(path: str | Path, *, max_bytes: int = 20_000_000) -> list[Page]:
    file = Path(path)
    if not file.is_file() or file.suffix.lower() not in {".pdf", ".txt"}:
        raise ValueError("Only existing PDF and TXT files are supported")
    if file.stat().st_size > max_bytes:
        raise ValueError("Document exceeds the configured size limit")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    if file.suffix.lower() == ".pdf":
        pages = [
            Page((p.extract_text() or "").strip(), file.name, digest, i)
            for i, p in enumerate(PdfReader(file).pages, 1)
        ]
    else:
        text = file.read_text(encoding="utf-8", errors="replace").strip()
        pages = [Page(text, file.name, digest, 1)]
    pages = [p for p in pages if p.text]
    if not pages:
        raise ValueError("Document contains no extractable text")
    return pages
