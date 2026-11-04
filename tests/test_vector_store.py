import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from genai_engineering.core.embeddings import MockEmbeddingProvider
from genai_engineering.core.errors import EmbeddingCompatibilityError
from genai_engineering.task2_rag.vector_store import ChromaStore


def test_chroma_persistence_api_add_search_and_delete(tmp_path):
    embedder = MockEmbeddingProvider(dimension=2)
    store = ChromaStore(str(tmp_path / "vectors"), "test_docs", embedder)
    chunk = {
        "id": "hash:1:0",
        "text": "The answer is forty two.",
        "metadata": {"source": "guide.txt", "source_id": "hash", "page": 1},
    }
    assert store.add([chunk], [[1.0, 0.0]]) == 1
    assert store.add([chunk], [[1.0, 0.0]]) == 0
    hits = store.search([1.0, 0.0], top_k=1)
    assert hits[0]["text"] == chunk["text"]
    assert hits[0]["metadata"]["page"] == 1
    store.delete_source("hash")
    assert store.search([1.0, 0.0], top_k=1) == []


def test_collection_rejects_embedding_model_change(tmp_path):
    path = str(tmp_path / "vectors")
    first = MockEmbeddingProvider(dimension=2, model_name="model-a")
    store = ChromaStore(path, "docs", first)
    chunk = {
        "id": "one",
        "text": "stored text",
        "metadata": {"source": "a.txt", "source_id": "a", "page": 1},
    }
    store.add([chunk], [[1.0, 0.0]])

    with pytest.raises(EmbeddingCompatibilityError, match="incompatible embeddings"):
        ChromaStore(
            path, "docs", MockEmbeddingProvider(dimension=2, model_name="model-b")
        )


def test_collection_rejects_legacy_vectors_without_fingerprint(tmp_path):
    path = str(tmp_path / "vectors")
    client = chromadb.PersistentClient(
        path=path, settings=ChromaSettings(anonymized_telemetry=False)
    )
    legacy = client.get_or_create_collection(
        "legacy", metadata={"hnsw:space": "cosine"}
    )
    legacy.add(
        ids=["legacy-1"],
        documents=["old document"],
        metadatas=[{"source": "old.txt"}],
        embeddings=[[1.0, 0.0]],
    )
    with pytest.raises(EmbeddingCompatibilityError, match="legacy data"):
        ChromaStore(path, "legacy", MockEmbeddingProvider(dimension=2))
    assert legacy.count() == 1


def test_dimension_is_checked_before_vector_database_write(tmp_path):
    store = ChromaStore(
        str(tmp_path / "vectors"), "docs", MockEmbeddingProvider(dimension=2)
    )
    chunk = {"id": "x", "text": "x", "metadata": {"source": "x.txt"}}
    with pytest.raises(EmbeddingCompatibilityError, match="expected 2, got 1"):
        store.add([chunk], [[1.0]])
