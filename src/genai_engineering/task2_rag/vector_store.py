from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from genai_engineering.core.embeddings import EmbeddingProvider
from genai_engineering.core.errors import EmbeddingCompatibilityError


class ChromaStore:
    def __init__(
        self, path: str, collection: str, embeddings: EmbeddingProvider
    ) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            collection, metadata=self._expected_metadata()
        )
        self._validate_collection_metadata()

    def _expected_metadata(self) -> dict[str, str | int]:
        return {
            "hnsw:space": "cosine",
            "embedding_provider": self.embeddings.provider_name,
            "embedding_model": self.embeddings.get_model_name(),
            "embedding_revision": self.embeddings.get_model_revision(),
            "embedding_dimension": self.embeddings.get_dimension(),
            "embedding_metadata_version": 1,
        }

    def _validate_collection_metadata(self) -> None:
        expected = self._expected_metadata()
        actual = self.collection.metadata or {}
        fingerprint_keys = (
            "embedding_provider",
            "embedding_model",
            "embedding_revision",
            "embedding_dimension",
            "embedding_metadata_version",
        )
        if any(key not in actual for key in fingerprint_keys):
            raise EmbeddingCompatibilityError(
                f"Collection {self.collection.name!r} has no complete embedding "
                "fingerprint. It is treated as legacy data; re-embed its stored "
                "documents into a new collection before use."
            )
        mismatches = {
            key: (actual.get(key), expected[key])
            for key in fingerprint_keys
            if actual[key] != expected[key]
        }
        if mismatches:
            details = ", ".join(
                f"{key}: stored={stored!r}, configured={configured!r}"
                for key, (stored, configured) in mismatches.items()
            )
            raise EmbeddingCompatibilityError(
                f"Collection {self.collection.name!r} uses incompatible embeddings "
                f"({details}). Use a new collection or run the documented migration."
            )
        distance = actual.get("hnsw:space", "l2")
        if distance != "cosine":
            raise EmbeddingCompatibilityError(
                f"Collection {self.collection.name!r} uses distance function "
                f"{distance!r}; NexaMind requires 'cosine'. Create a fresh collection."
            )

    def _validate_vector(self, vector: list[float]) -> None:
        expected = self.embeddings.get_dimension()
        if len(vector) != expected:
            raise EmbeddingCompatibilityError(
                f"Vector dimension mismatch: expected {expected}, got {len(vector)}"
            )

    def new_chunks(self, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        existing = set(self.collection.get(ids=[c["id"] for c in chunks])["ids"])
        return [c for c in chunks if c["id"] not in existing]

    def add(self, chunks: list[dict], vectors: list[list[float]]) -> int:
        if not chunks:
            return 0
        for vector in vectors:
            self._validate_vector(vector)
        new_ids = {chunk["id"] for chunk in self.new_chunks(chunks)}
        pairs = [
            (chunk, vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
            if chunk["id"] in new_ids
        ]
        if pairs:
            self.collection.add(
                ids=[c["id"] for c, _ in pairs],
                documents=[c["text"] for c, _ in pairs],
                metadatas=[c["metadata"] for c, _ in pairs],
                embeddings=[v for _, v in pairs],
            )
        return len(pairs)

    def search(
        self, vector: list[float], *, top_k: int = 4, where: dict | None = None
    ) -> list[dict]:
        self._validate_vector(vector)
        self._validate_collection_metadata()
        result = self.collection.query(
            query_embeddings=[vector], n_results=top_k, where=where
        )
        hits = []
        for i, text in enumerate(result["documents"][0]):
            hits.append(
                {
                    "text": text,
                    "metadata": result["metadatas"][0][i],
                    "score": 1 - result["distances"][0][i],
                }
            )
        return hits

    def delete_source(self, source_id: str) -> None:
        self.collection.delete(where={"source_id": source_id})
