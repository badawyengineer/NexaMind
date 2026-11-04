"""Deterministic retrieval and citation checks; no model calls are made."""


def retrieval_relevance(
    hits: list[dict], expected_source: str, expected_page: int
) -> float:
    if not hits:
        return 0.0
    return sum(
        1
        for hit in hits
        if hit["metadata"].get("source") == expected_source
        and hit["metadata"].get("page") == expected_page
    ) / len(hits)


def context_recall(
    hits: list[dict], relevant_source_pages: set[tuple[str, int]]
) -> float:
    if not relevant_source_pages:
        return 1.0
    found = {(h["metadata"].get("source"), h["metadata"].get("page")) for h in hits}
    return len(found & relevant_source_pages) / len(relevant_source_pages)


def citation_correctness(
    sources: list[dict], indexed_pages: set[tuple[str, int]]
) -> bool:
    return all((s.get("source"), s.get("page")) in indexed_pages for s in sources)
