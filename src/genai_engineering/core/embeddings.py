"""Provider-independent interfaces and adapters for document embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from genai_engineering.core.config import Settings, get_settings
from genai_engineering.core.errors import ConfigurationError, ProviderError
from genai_engineering.core.observability import observe


class EmbeddingProvider(Protocol):
    """Common vector contract used by ingestion and retrieval."""

    provider_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def get_dimension(self) -> int: ...

    def get_model_name(self) -> str: ...

    def get_model_revision(self) -> str: ...


class _ValidatedEmbeddingProvider:
    provider_name: str

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.embedding_model
        self.model_revision = settings.embedding_revision
        self._dimension = settings.embedding_dimensions

    def get_dimension(self) -> int:
        if self._dimension is None:
            raise ConfigurationError(
                f"Embedding dimension for {self.model_name!r} is unknown; "
                "set EMBEDDING_DIMENSIONS or use a supported model"
            )
        return self._dimension

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_revision(self) -> str:
        return self.model_revision

    def _validate(
        self, texts: Sequence[str], vectors: Sequence[Sequence[float]]
    ) -> list[list[float]]:
        if len(texts) != len(vectors):
            raise ProviderError(
                "Embedding provider returned an unexpected vector count"
            )
        normalized = [[float(value) for value in vector] for vector in vectors]
        dimensions = {len(vector) for vector in normalized}
        if len(dimensions) > 1:
            raise ProviderError("Embedding provider returned inconsistent dimensions")
        if dimensions:
            actual = next(iter(dimensions))
            if actual <= 0:
                raise ProviderError("Embedding provider returned an empty vector")
            if self._dimension is not None and actual != self._dimension:
                raise ProviderError(
                    "Embedding dimension mismatch: "
                    f"expected {self._dimension}, got {actual}"
                )
            self._dimension = actual
        return normalized


class OpenAIEmbeddingProvider(_ValidatedEmbeddingProvider):
    provider_name = "openai"
    _known_dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for OpenAI embeddings")
        if self._dimension is None:
            self._dimension = self._known_dimensions.get(self.model_name)
        if self._dimension is None:
            raise ConfigurationError(
                f"Unknown dimension for OpenAI model {self.model_name!r}; "
                "set EMBEDDING_DIMENSIONS"
            )
        from openai import OpenAI

        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.settings.embedding_batch_size):
            batch = texts[start : start + self.settings.embedding_batch_size]
            kwargs = {"model": self.model_name, "input": batch}
            if self.settings.embedding_dimensions is not None:
                kwargs["dimensions"] = self.settings.embedding_dimensions
            with observe(
                "embedding.request",
                provider=self.provider_name,
                model=self.model_name,
                item_count=len(batch),
            ) as event:
                result = self.client.embeddings.create(**kwargs)
                if getattr(result, "usage", None):
                    event["input_tokens"] = result.usage.prompt_tokens
            vectors.extend(item.embedding for item in result.data)
        return self._validate(texts, vectors)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embed(texts)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenAI document embedding failed") from exc

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Query text must not be empty")
        return self.embed_documents([text])[0]


class GeminiEmbeddingProvider(_ValidatedEmbeddingProvider):
    provider_name = "gemini"
    _known_dimensions = {"gemini-embedding-001": 3072, "gemini-embedding-2": 3072}

    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.gemini_api_key:
            raise ConfigurationError("GEMINI_API_KEY is required for Gemini embeddings")
        if self._dimension is None:
            self._dimension = self._known_dimensions.get(self.model_name)
        if self._dimension is None:
            raise ConfigurationError(
                f"Unknown dimension for Gemini model {self.model_name!r}; "
                "set EMBEDDING_DIMENSIONS"
            )
        from google import genai

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options={"timeout": int(settings.request_timeout * 1000)},
        )

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.settings.embedding_batch_size):
            batch = texts[start : start + self.settings.embedding_batch_size]
            config = types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.settings.embedding_dimensions,
            )
            with observe(
                "embedding.request",
                provider=self.provider_name,
                model=self.model_name,
                item_count=len(batch),
            ):
                result = self.client.models.embed_content(
                    model=self.model_name, contents=batch, config=config
                )
            vectors.extend(embedding.values for embedding in result.embeddings or [])
        return self._validate(texts, vectors)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embed(texts, "RETRIEVAL_DOCUMENT")
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Gemini document embedding failed") from exc

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Query text must not be empty")
        try:
            return self._embed([text], "RETRIEVAL_QUERY")[0]
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Gemini query embedding failed") from exc


class SentenceTransformersEmbeddingProvider(_ValidatedEmbeddingProvider):
    provider_name = "sentence_transformers"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError(
                "Install local embeddings with `uv sync --extra local-embeddings`"
            ) from exc
        try:
            self.model = SentenceTransformer(
                self.model_name,
                revision=None
                if self.model_revision == "default"
                else self.model_revision,
                device=settings.embedding_device,
            )
            model_dimension = self.model.get_sentence_embedding_dimension()
            if self._dimension is not None and model_dimension != self._dimension:
                raise ProviderError(
                    f"Configured embedding dimension {self._dimension} does not match "
                    f"model dimension {model_dimension}"
                )
            self._dimension = model_dimension
        except (ConfigurationError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderError(
                "Could not load the Sentence Transformers model"
            ) from exc

    def _encode(self, texts: list[str], *, query: bool) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), self.settings.embedding_batch_size):
                batch = texts[start : start + self.settings.embedding_batch_size]
                encoder = (
                    getattr(self.model, "encode_query", None)
                    if query
                    else getattr(self.model, "encode_document", None)
                )
                if encoder is None:
                    encoder = self.model.encode
                encoded = encoder(batch, convert_to_numpy=True, show_progress_bar=False)
                vectors.extend(
                    vector.tolist() if hasattr(vector, "tolist") else list(vector)
                    for vector in encoded
                )
            return self._validate(texts, vectors)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Sentence Transformers embedding failed") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, query=False)

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Query text must not be empty")
        return self._encode([text], query=True)[0]


def create_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    factories = {
        "openai": OpenAIEmbeddingProvider,
        "gemini": GeminiEmbeddingProvider,
        "sentence_transformers": SentenceTransformersEmbeddingProvider,
    }
    try:
        factory = factories[settings.embedding_provider]
    except KeyError as exc:
        raise ConfigurationError(
            f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}"
        ) from exc
    model_defaults = {
        "gemini": "gemini-embedding-001",
        "sentence_transformers": "sentence-transformers/all-MiniLM-L6-v2",
    }
    if (
        settings.embedding_provider in model_defaults
        and settings.embedding_model == "text-embedding-3-small"
    ):
        settings = settings.model_copy(
            update={"embedding_model": model_defaults[settings.embedding_provider]}
        )
    return factory(settings)


class MockEmbeddingProvider(_ValidatedEmbeddingProvider):
    """Small deterministic provider for unit tests and local examples."""

    provider_name = "mock"

    def __init__(self, dimension: int = 4, model_name: str = "mock-embedding-v1"):
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.settings = Settings(
            embedding_model=model_name, embedding_dimensions=dimension
        )
        self.model_name = model_name
        self.model_revision = "test-fixture"
        self._dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            vector = [0.0] * self.get_dimension()
            for index, byte in enumerate(text.encode("utf-8")):
                vector[index % len(vector)] += float(byte)
            result.append(vector)
        return self._validate(texts, result)

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Query text must not be empty")
        return self.embed_documents([text])[0]
