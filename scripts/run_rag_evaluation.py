"""Run a retrieval configuration matrix over the curated RAG cases."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from genai_engineering.core.config import get_settings
from genai_engineering.core.embeddings import create_embedding_provider
from genai_engineering.core.providers import create_provider
from genai_engineering.task2_rag.chunking import split_pages
from genai_engineering.task2_rag.ingestion import load_document
from genai_engineering.task2_rag.service import RAGService
from genai_engineering.task2_rag.vector_store import ChromaStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/evaluation/rag_questions.json")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[500, 900])
    parser.add_argument("--overlaps", nargs="+", type=int, default=[80, 120])
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--store-path", default="data/evaluation/vectorstore")
    parser.add_argument("--output", default="reports/evaluation/run-matrix.json")
    parser.add_argument(
        "--append",
        action="store_true",
        help="append configurations to an existing output file",
    )
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help=(
            "also call the configured LLM for every case/configuration "
            "(may incur charges)"
        ),
    )
    args = parser.parse_args()
    if any(size <= 0 for size in args.chunk_sizes):
        parser.error("chunk sizes must be positive")
    if any(overlap < 0 for overlap in args.overlaps):
        parser.error("overlaps cannot be negative")
    if any(size <= overlap for size in args.chunk_sizes for overlap in args.overlaps):
        parser.error("every chunk size must be larger than every overlap")
    if any(top_k < 1 or top_k > 20 for top_k in args.top_k_values):
        parser.error("top-k values must be 1..20")
    if any(not -1 <= threshold <= 1 for threshold in args.thresholds):
        parser.error("threshold values must be between -1 and 1")
    settings = get_settings()
    embedding_provider = create_embedding_provider(settings)
    llm_provider = create_provider(settings) if args.generate_answers else None
    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    sources = sorted(Path("data/samples").glob("*.txt")) + sorted(
        Path("data/evaluation/fixtures").glob("*.txt")
    )
    pages = []
    for source in sources:
        pages.extend(
            load_document(
                source,
                max_bytes=settings.max_file_bytes,
                max_pages=settings.max_document_pages,
                max_image_pixels=settings.max_image_pixels,
            )
        )
    query_vectors = {
        case["id"]: embedding_provider.embed_query(case["question"]) for case in cases
    }

    runs = []
    provider_tag = re.sub(r"[^a-z0-9]", "", settings.embedding_provider.lower())[:12]
    embedding_model_name = embedding_provider.get_model_name()
    model_tag = hashlib.sha1(embedding_model_name.encode()).hexdigest()[:6]
    for chunk_size in args.chunk_sizes:
        for overlap in args.overlaps:
            collection = f"eval_{provider_tag}_{model_tag}_{chunk_size}_{overlap}"
            store = ChromaStore(args.store_path, collection, embedding_provider)
            chunks = split_pages(pages, chunk_size=chunk_size, overlap=overlap)
            chunks = store.new_chunks(chunks)
            if chunks:
                vectors = embedding_provider.embed_documents(
                    [chunk["text"] for chunk in chunks]
                )
                store.add(chunks, vectors)
            service = (
                RAGService(llm_provider, embedding_provider, store)
                if args.generate_answers
                else None
            )
            max_k = max(args.top_k_values)
            candidates_by_case = {
                case["id"]: store.search(query_vectors[case["id"]], top_k=max_k)
                for case in cases
            }
            for top_k in args.top_k_values:
                for threshold in args.thresholds:
                    results = []
                    for case in cases:
                        hits = [
                            item
                            for item in candidates_by_case[case["id"]][:top_k]
                            if item["score"] >= threshold
                        ]
                        answer = (
                            service.answer_from_hits(case["question"], hits)
                            if args.generate_answers
                            else {"answer": "", "sources": []}
                        )
                        results.append(
                            {
                                "case_id": case["id"],
                                "hits": hits,
                                "answer": answer["answer"],
                                "sources": answer["sources"],
                            }
                        )
                    runs.append(
                        {
                            "config": {
                                "provider": settings.embedding_provider,
                                "model": embedding_model_name,
                                "llm_provider": settings.llm_provider,
                                "llm_model": settings.model,
                                "chunk_size": chunk_size,
                                "overlap": overlap,
                                "top_k": top_k,
                                "threshold": threshold,
                                "answers_generated": args.generate_answers,
                            },
                            "results": results,
                        }
                    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.append and output.exists():
        runs = json.loads(output.read_text(encoding="utf-8")) + runs
    output.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(runs)} configurations to {output}")


if __name__ == "__main__":
    main()
