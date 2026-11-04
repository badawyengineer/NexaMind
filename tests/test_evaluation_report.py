from genai_engineering.task2_rag.reporting import build_report, render_markdown


def test_report_compares_saved_configuration_runs():
    dataset = [
        {
            "id": "case-1",
            "question": "total?",
            "reference_answer": "13.20",
            "relevant_chunk_ids": ["chunk-1"],
            "relevant_pages": [{"source": "invoice.pdf", "page": 1}],
        }
    ]
    runs = [
        {
            "config": {"provider": "local", "model": "small", "top_k": 2},
            "results": [
                {
                    "case_id": "case-1",
                    "hits": [
                        {
                            "id": "chunk-1",
                            "text": "Total USD 13.20",
                            "metadata": {
                                "chunk_id": "chunk-1",
                                "source": "invoice.pdf",
                                "page": 1,
                            },
                        }
                    ],
                    "answer": "The total is USD 13.20 [invoice.pdf, p. 1].",
                    "sources": [{"source": "invoice.pdf", "page": 1}],
                }
            ],
        }
    ]

    report = build_report(dataset, runs)
    markdown = render_markdown(report)

    assert report["runs"][0]["means"]["mrr"] == 1.0
    assert report["runs"][0]["cases"][0]["has_citation"]
    assert "local / small" in markdown
