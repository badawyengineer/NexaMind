"""Re-embed a legacy Chroma collection into a new fingerprinted collection."""

import argparse

import chromadb
from chromadb.config import Settings as ChromaSettings

from genai_engineering.core.config import get_settings
from genai_engineering.core.embeddings import create_embedding_provider
from genai_engineering.task2_rag.vector_store import ChromaStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-collection", required=True)
    parser.add_argument("--target-collection", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 512:
        parser.error("--batch-size must be between 1 and 512")
    if args.source_collection == args.target_collection:
        parser.error("Source and target collection names must differ")

    settings = get_settings()
    client = chromadb.PersistentClient(
        path=settings.chroma_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    names = {collection.name for collection in client.list_collections()}
    if args.source_collection not in names:
        parser.error(f"Source collection {args.source_collection!r} does not exist")
    if args.target_collection in names:
        parser.error(
            f"Target collection {args.target_collection!r} already exists; "
            "choose a fresh name so the source remains untouched"
        )

    source = client.get_collection(args.source_collection)
    provider = create_embedding_provider(settings)
    target = ChromaStore(settings.chroma_path, args.target_collection, provider)
    total = source.count()
    for offset in range(0, total, args.batch_size):
        batch = source.get(
            limit=args.batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        documents = batch.get("documents") or []
        metadata = batch.get("metadatas") or [{} for _ in documents]
        ids = batch.get("ids") or []
        chunks = [
            {
                "id": chunk_id,
                "text": document,
                "metadata": item_metadata or {"source": "legacy-unknown"},
            }
            for chunk_id, document, item_metadata in zip(
                ids, documents, metadata, strict=True
            )
        ]
        vectors = provider.embed_documents([chunk["text"] for chunk in chunks])
        target.add(chunks, vectors)
        print(f"Re-embedded {min(offset + len(chunks), total)}/{total} chunks")
    print(
        f"Migration complete. Verify collection {args.target_collection!r}, then "
        "set CHROMA_COLLECTION to that name. The source was left unchanged."
    )


if __name__ == "__main__":
    main()
