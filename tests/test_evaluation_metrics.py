from genai_engineering.task2_rag.evaluation import (
    hit_rate,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    score_case,
)


def test_rank_metrics_use_exact_relevant_chunk_ids():
    hits = [
        {"id": "wrong", "metadata": {"chunk_id": "wrong"}},
        {"id": "right", "metadata": {"chunk_id": "right"}},
    ]
    relevant = {"right"}

    assert precision_at_k(hits, relevant, 2) == 0.5
    assert recall_at_k(hits, relevant, 2) == 1.0
    assert mean_reciprocal_rank(hits, relevant) == 0.5
    assert hit_rate(hits, relevant, 1) == 0.0
    assert precision_at_k(hits[:1], relevant, 5) == 0.0


def test_case_scoring_checks_abstention_and_labels_lexical_proxies():
    case = {
        "id": "unsupported",
        "question": "Which medicine?",
        "reference_answer": "",
        "should_abstain": True,
    }
    result = score_case(
        case,
        [],
        "I couldn't find enough information in the indexed documents to answer that.",
    )

    assert result["abstention_correct"] is True
    assert "not semantic" in result["evaluation_note"]
