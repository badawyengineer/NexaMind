# RAG evaluation

`data/evaluation/rag_questions.json` contains direct lookup, combined-fact,
unsupported, misleading-instruction, and conflicting-evidence cases. Fixture
documents live beside the dataset. Chunk IDs in each case identify expected
relevant chunks; source/page labels support context recall.

Run the retrieval matrix after configuring `EMBEDDING_PROVIDER` and model settings:

```powershell
uv run python scripts/run_rag_evaluation.py `
  --chunk-sizes 500 900 `
  --overlaps 80 120 `
  --top-k-values 3 5 `
  --thresholds 0.1 0.2
```

Remote embedding providers make API calls and may incur charges. Use the optional
Sentence Transformers provider for local embedding runs. The default matrix does
not generate answers; add `--generate-answers` to call the configured LLM once for
each case/configuration. Run again with other provider/model settings to compare
providers, using `--append` with the same output path to retain earlier runs. The
script stores vectors under `data/evaluation/vectorstore` and
produces a JSON matrix under `reports/evaluation`.

Score saved runs and make a Markdown comparison table:

```powershell
uv run python scripts/report_evaluation.py `
  --results reports/evaluation/run-matrix.json `
  --output-dir reports/evaluation/scored
```

Each run records provider, model, chunk size, overlap, top-k, threshold, hits,
answers, and sources. The JSON report includes precision@K, recall@K, MRR, hit
rate, source/page context recall, lexical query/context overlap, reference answer
token coverage, answer/context token support, required-phrase support, abstention,
and citation checks. Token and phrase metrics are deterministic proxies; human
review is still needed for semantic correctness, contradiction handling, and
faithfulness. Do not describe lexical overlap as an LLM judge score.

To exercise the real OCR binary against a generated scanned-PDF fixture, install
Tesseract plus English data, then run the opt-in test:

```powershell
$env:RUN_OCR_TESTS = "1"
uv run pytest tests/test_ocr_live.py -q
```

This check is skipped by default and is not part of normal CI.
