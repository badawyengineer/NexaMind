"""Backward-compatible imports for embedding adapters."""

from genai_engineering.core.embeddings import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformersEmbeddingProvider,
    create_embedding_provider,
)

# The former module exposed this OpenAI-only concrete class as ``Embeddings``.
Embeddings = OpenAIEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "Embeddings",
    "GeminiEmbeddingProvider",
    "MockEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformersEmbeddingProvider",
    "create_embedding_provider",
]
