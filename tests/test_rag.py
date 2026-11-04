from genai_engineering.task2_rag.chunking import split_pages
from genai_engineering.task2_rag.evaluation import (
    citation_correctness,
    context_recall,
    retrieval_relevance,
)
from genai_engineering.task2_rag.ingestion import Page
from genai_engineering.task2_rag.service import RAGService


def test_chunks_keep_page_metadata():
    page = Page("Important sentence. " * 80, "guide.pdf", "abc", 7)
    chunks = split_pages([page], chunk_size=150, overlap=20, min_length=10)
    assert chunks
    assert all(
        c["metadata"]["page"] == 7 and c["metadata"]["source"] == "guide.pdf"
        for c in chunks
    )
    assert all(c["metadata"]["chunk_id"] == c["id"] for c in chunks)


def test_no_evidence_returns_clear_message():
    class Embed:
        def embed(self, texts):
            return [[0.1]]

    class Store:
        def search(self, vector, **kwargs):
            return []

    service = RAGService(object(), Embed(), Store())
    assert service.answer("unknown") == {
        "answer": (
            "I couldn't find enough information in the indexed documents "
            "to answer that."
        ),
        "sources": [],
    }


def test_deterministic_evaluation_checks_sources_and_pages():
    hits = [{"metadata": {"source": "guide.pdf", "page": 7}}]
    assert retrieval_relevance(hits, "guide.pdf", 7) == 1.0
    assert context_recall(hits, {("guide.pdf", 7)}) == 1.0
    assert citation_correctness(
        [{"source": "guide.pdf", "page": 7}], {("guide.pdf", 7)}
    )


def test_duplicate_ingestion_skips_embedding(tmp_path):
    file = tmp_path / "notes.txt"
    file.write_text("Useful content. " * 30, encoding="utf-8")

    class Embed:
        calls = 0

        def embed(self, texts):
            self.calls += 1
            return [[0.1] for _ in texts]

    class Store:
        known = set()

        def new_chunks(self, chunks):
            return [chunk for chunk in chunks if chunk["id"] not in self.known]

        def add(self, chunks, vectors):
            self.known.update(chunk["id"] for chunk in chunks)
            return len(chunks)

    embedding, store = Embed(), Store()
    service = RAGService(object(), embedding, store)
    assert service.ingest(str(file)) > 0
    assert service.ingest(str(file)) == 0
    assert embedding.calls == 1
