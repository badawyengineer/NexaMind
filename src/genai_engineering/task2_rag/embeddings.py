from genai_engineering.core.config import Settings, get_settings
from genai_engineering.core.errors import ConfigurationError, ProviderError


class Embeddings:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if self.settings.embedding_provider != "openai":
            raise ConfigurationError(
                "Only OpenAI embeddings are supported; set EMBEDDING_PROVIDER=openai"
            )
        if not self.settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for embeddings")
        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.request_timeout,
            max_retries=self.settings.max_retries,
        )
        self.dimensions: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = []
            for start in range(0, len(texts), 128):
                result = self.client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=texts[start : start + 128],
                )
                vectors.extend(item.embedding for item in result.data)
            sizes = {len(v) for v in vectors}
            if len(sizes) != 1 or (
                self.dimensions is not None and sizes != {self.dimensions}
            ):
                raise ValueError("Embedding dimensions changed")
            self.dimensions = next(iter(sizes))
            return vectors
        except Exception as exc:
            raise ProviderError("Embedding generation failed") from exc
