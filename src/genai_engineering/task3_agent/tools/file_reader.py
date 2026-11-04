from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

from genai_engineering.core.errors import ToolError


class FileReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=500)


class SafeFileReader:
    def __init__(self, root: str, max_bytes: int = 5_000_000, max_pages: int = 100):
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.max_pages = max_pages

    def read(self, relative_path: str) -> str:
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ToolError("Access denied: path is outside the allowed directory")
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            raise ToolError("Only TXT, Markdown and PDF files are allowed")
        if not path.is_file():
            raise ToolError("File not found")
        if path.stat().st_size > self.max_bytes:
            raise ToolError("File exceeds the size limit")
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(path)
            if len(reader.pages) > self.max_pages:
                raise ToolError("PDF exceeds the configured page limit")
            return "\n".join(page.extract_text() or "" for page in reader.pages)[
                :100_000
            ]
        return path.read_text(encoding="utf-8", errors="replace")[:100_000]
