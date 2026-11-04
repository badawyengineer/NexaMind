"""Score saved retrieval/answer runs and emit machine/human-readable reports."""

import argparse
import json
from pathlib import Path

from genai_engineering.task2_rag.evaluation import score_case


def build_report(dataset: list[dict], runs: list[dict]) -> dict:
    cases = {case["id"]: case for case in dataset}
    report_runs = []
    for run in runs:
        scored = []
        for result in run.get("results", []):
            case = cases[result["case_id"]]
            scored.append(
                score_case(
                    case,
                    result.get("hits", []),
                    result.get("answer", ""),
                    result.get("sources"),
                    run.get("config", {}).get("top_k"),
                )
            )
        metric_names = [
            "precision_at_k",
            "recall_at_k",
            "mrr",
            "hit_rate",
            "context_recall",
            "context_query_overlap",
            "reference_answer_token_coverage",
            "answer_context_token_support",
            "reference_context_token_support",
            "abstention_correct",
            "citations_match_retrieved_sources",
            "has_citation",
        ]
        means = {}
        for metric in metric_names:
            values = [item[metric] for item in scored if item[metric] is not None]
            means[metric] = sum(values) / len(values) if values else None
        report_runs.append({**run.get("config", {}), "means": means, "cases": scored})
    return {"runs": report_runs}


def render_markdown(report: dict) -> str:
    lines = [
        "# RAG evaluation report",
        "",
        "Lexical answer/context metrics are reproducible proxies, not semantic "
        "faithfulness judgments. Review individual cases and source citations.",
        "",
        "| Provider / model | Chunk size / overlap | Top K / threshold | MRR | "
        "Recall@K | Context recall | Answer token coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        means = run["means"]
        lines.append(
            "| {provider} / {model} | {chunk_size} / {overlap} | {top_k} / "
            "{threshold} | {mrr} | {recall} | {context} | {answer} |".format(
                provider=run.get("provider", "unspecified"),
                model=run.get("model", "unspecified"),
                chunk_size=run.get("chunk_size", "unspecified"),
                overlap=run.get("overlap", "unspecified"),
                top_k=run.get("top_k", "unspecified"),
                threshold=run.get("threshold", "unspecified"),
                mrr=_fmt(means["mrr"]),
                recall=_fmt(means["recall_at_k"]),
                context=_fmt(means["context_recall"]),
                answer=_fmt(means["reference_answer_token_coverage"]),
            )
        )
    lines.extend(["", "## Per-case results", ""])
    for run_index, run in enumerate(report["runs"], 1):
        lines.extend(
            [
                f"### Run {run_index}",
                "",
                "| Case | MRR | Recall@K | Abstention correct |",
                "|---|---:|---:|---|",
            ]
        )
        for item in run["cases"]:
            lines.append(
                f"| {item['case_id']} | {_fmt(item['mrr'])} | "
                f"{_fmt(item['recall_at_k'])} | {item['abstention_correct']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/evaluation/rag_questions.json")
    parser.add_argument("--results", required=True, help="Saved run matrix JSON")
    parser.add_argument("--output-dir", default="reports/evaluation")
    args = parser.parse_args()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    runs = json.loads(Path(args.results).read_text(encoding="utf-8"))
    report = build_report(dataset, runs)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "evaluation.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote evaluation reports to {output}")


if __name__ == "__main__":
    main()
