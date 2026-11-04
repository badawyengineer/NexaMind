import sys
from types import ModuleType, SimpleNamespace

import pytest

from genai_engineering.core.config import Settings
from genai_engineering.core.embeddings import (
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformersEmbeddingProvider,
    create_embedding_provider,
)
from genai_engineering.core.errors import ProviderError


def test_mock_provider_is_deterministic():
    provider = MockEmbeddingProvider(dimension=3)
    assert provider.embed_documents(["abc"]) == provider.embed_documents(["abc"])
    assert len(provider.embed_query("query")) == 3


def test_openai_provider_batches_and_checks_dimensions(monkeypatch):
    import openai

    batches = []

    class Client:
        class embeddings:
            @staticmethod
            def create(**kwargs):
                batches.append(kwargs)
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=[1.0, 2.0]) for _ in kwargs["input"]
                    ]
                )

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: Client())
    settings = Settings(
        openai_api_key="test-key",
        embedding_dimensions=2,
        embedding_batch_size=2,
    )
    provider = OpenAIEmbeddingProvider(settings)
    vectors = provider.embed_documents(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert [len(batch["input"]) for batch in batches] == [2, 2, 1]
    assert provider.get_dimension() == 2

    class WrongClient:
        class embeddings:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])])

    provider.client = WrongClient()
    with pytest.raises(ProviderError, match="dimension mismatch"):
        provider.embed_documents(["bad dimension"])


def test_gemini_provider_uses_retrieval_task_types(monkeypatch):
    from google import genai

    tasks = []

    class Models:
        @staticmethod
        def embed_content(*, model, contents, config):
            tasks.append(config.task_type)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.25, 0.75]) for _ in contents]
            )

    monkeypatch.setattr(
        genai,
        "Client",
        lambda **kwargs: SimpleNamespace(models=Models()),
    )
    provider = GeminiEmbeddingProvider(
        Settings(
            llm_provider="gemini",
            gemini_api_key="test-key",
            embedding_provider="gemini",
            embedding_model="gemini-embedding-001",
            embedding_dimensions=2,
        )
    )
    assert provider.embed_documents(["document"])[0] == [0.25, 0.75]
    assert provider.embed_query("question") == [0.25, 0.75]
    assert tasks == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


def test_sentence_transformer_adapter_uses_query_and_document_methods(monkeypatch):
    calls = []

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def get_sentence_embedding_dimension(self):
            return 2

        def encode_document(self, texts, **kwargs):
            calls.append("document")
            return [[1.0, 0.0] for _ in texts]

        def encode_query(self, texts, **kwargs):
            calls.append("query")
            return [[0.0, 1.0] for _ in texts]

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    provider = SentenceTransformersEmbeddingProvider(
        Settings(
            embedding_provider="sentence_transformers",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    assert provider.embed_documents(["doc"]) == [[1.0, 0.0]]
    assert provider.embed_query("query") == [0.0, 1.0]
    assert calls == ["document", "query"]


def test_factory_selects_provider_specific_default_models(monkeypatch):
    class FakeProvider:
        def __init__(self, settings):
            self.settings = settings

    import genai_engineering.core.embeddings as embeddings

    monkeypatch.setattr(embeddings, "GeminiEmbeddingProvider", FakeProvider)
    provider = create_embedding_provider(
        Settings(
            embedding_provider="gemini",
            gemini_api_key="test-key",
        )
    )
    assert provider.settings.embedding_model == "gemini-embedding-001"
