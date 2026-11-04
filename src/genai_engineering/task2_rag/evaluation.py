"""Deterministic, lexical evaluation helpers; no judge model calls are made."""

import re


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _hit_id(hit: dict) -> str | None:
    return hit.get("metadata", {}).get("chunk_id") or hit.get("id")


def precision_at_k(hits: list[dict], relevant_ids: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    selected = hits[:k]
    return sum(_hit_id(hit) in relevant_ids for hit in selected) / k


def recall_at_k(hits: list[dict], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    found = {_hit_id(hit) for hit in hits[:k]}
    return len(found & relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(hits: list[dict], relevant_ids: set[str]) -> float:
    for rank, hit in enumerate(hits, 1):
        if _hit_id(hit) in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_rate(hits: list[dict], relevant_ids: set[str], k: int) -> float:
    return float(any(_hit_id(hit) in relevant_ids for hit in hits[:k]))


def lexical_overlap(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


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


def score_case(
    case: dict,
    hits: list[dict],
    answer: str,
    sources: list[dict] | None = None,
    k: int | None = None,
) -> dict:
    """Score one case; lexical answer checks are proxies for human judgment."""
    relevant_ids = set(case.get("relevant_chunk_ids", []))
    relevant_pages = {
        (item["source"], item["page"]) for item in case.get("relevant_pages", [])
    }
    context = "\n".join(hit.get("text", "") for hit in hits)
    expected = case.get("reference_answer", "")
    answer_tokens = _tokens(answer)
    expected_tokens = _tokens(expected)
    context_tokens = _tokens(context)
    rank_k = k or max(len(hits), 1)
    reference_coverage = (
        len(answer_tokens & expected_tokens) / len(expected_tokens)
        if expected_tokens
        else 1.0
    )
    answer_context_support = (
        len(answer_tokens & context_tokens) / len(answer_tokens)
        if answer_tokens
        else 1.0
    )
    reference_context_support = (
        len(expected_tokens & context_tokens) / len(expected_tokens)
        if expected_tokens
        else None
    )
    required = case.get("required_phrases", [])
    phrase_support = (
        sum(phrase.casefold() in context.casefold() for phrase in required)
        / len(required)
        if required
        else None
    )
    should_abstain = bool(case.get("should_abstain", False))
    abstained = any(
        phrase in answer.casefold()
        for phrase in ("couldn't find enough information", "insufficient information")
    )
    cited_pages = {
        (source, int(page))
        for source, page in re.findall(r"\[([^,\]]+),\s*p\.\s*(\d+)\]", answer)
    }
    retrieved_pages = {
        (item.get("source"), item.get("page"))
        for item in (sources or [hit.get("metadata", {}) for hit in hits])
    }
    return {
        "case_id": case["id"],
        "precision_at_k": (
            precision_at_k(hits, relevant_ids, rank_k) if relevant_ids else None
        ),
        "recall_at_k": (
            recall_at_k(hits, relevant_ids, rank_k) if relevant_ids else None
        ),
        "mrr": mean_reciprocal_rank(hits, relevant_ids) if relevant_ids else None,
        "hit_rate": hit_rate(hits, relevant_ids, rank_k) if relevant_ids else None,
        "context_recall": context_recall(hits, relevant_pages),
        "context_query_overlap": lexical_overlap(case["question"], context),
        "reference_answer_token_coverage": reference_coverage,
        "answer_context_token_support": answer_context_support,
        "reference_context_token_support": reference_context_support,
        "required_phrase_context_support": phrase_support,
        "has_citation": bool(cited_pages),
        "citations_match_retrieved_sources": cited_pages <= retrieved_pages,
        "abstention_expected": should_abstain,
        "abstention_correct": abstained == should_abstain,
        "evaluation_note": (
            "Token and phrase checks are reproducible proxies, not semantic "
            "faithfulness judgments. Review answer grounding manually."
        ),
    }
