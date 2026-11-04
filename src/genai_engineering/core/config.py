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
    chroma_path: str = "data/vectorstore"
    chroma_collection: str = "genai_documents"
    allowed_file_root: str = "data/samples"
    max_file_bytes: int = 5_000_000
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    request_timeout: float = Field(default=45.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)

    @property
    def model(self) -> str:
        return self.openai_model if self.llm_provider == "openai" else self.gemini_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
