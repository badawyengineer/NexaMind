"""Environment-backed configuration; credentials are never logged."""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "openai"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.0-flash"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = Field(default=None, gt=0)
    embedding_batch_size: int = Field(default=128, ge=1, le=1024)
    embedding_revision: str = "default"
    embedding_device: str = "cpu"
    chroma_path: str = "data/vectorstore"
    chroma_collection: str = "genai_documents"
    allowed_file_root: str = "data/samples"
    max_file_bytes: int = 5_000_000
    max_document_pages: int = Field(default=100, ge=1, le=1000)
    max_image_pixels: int = Field(default=40_000_000, ge=1_000_000)
    ocr_enabled: bool = False
    ocr_language: str = "eng"
    ocr_fallback_threshold: int = Field(default=20, ge=0)
    ocr_dpi: int = Field(default=200, ge=72, le=400)
    tesseract_cmd: str | None = None
    chat_persistence_enabled: bool = False
    chat_database_path: str = "data/conversations.sqlite3"
    chat_owner_header: str | None = None
    chat_session_secret: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    request_timeout: float = Field(default=45.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)

    @property
    def model(self) -> str:
        return self.openai_model if self.llm_provider == "openai" else self.gemini_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
