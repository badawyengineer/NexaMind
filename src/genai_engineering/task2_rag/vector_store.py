from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


class ChromaStore:
    def __init__(self, path: str, collection: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def new_chunks(self, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []
        existing = set(self.collection.get(ids=[c["id"] for c in chunks])["ids"])
        return [c for c in chunks if c["id"] not in existing]

    def add(self, chunks: list[dict], vectors: list[list[float]]) -> int:
        if not chunks:
            return 0
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
