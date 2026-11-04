# Engineering report

Date: 2026-11-04

## Delivered

- Task 1: invoice and resume Pydantic v2 schemas, parameterized prompts, OpenAI/Gemini text adapters, JSON validation, bounded repair, serialization, and sample input/output files.
- Task 2: TXT/PDF text ingestion, page-level metadata, recursive chunking, OpenAI batched embeddings, persistent Chroma storage, duplicate avoidance, filtered similarity retrieval, grounded generation, citations, and deterministic evaluation helpers.
- Task 3: native OpenAI function-call orchestration over web search, AST calculator, and confined local reader, with Pydantic argument schemas, iteration limits, observations, and trace display.
- Task 4: Streamlit extraction, RAG, and agent modes; session-scoped conversations; uploads; source citations; tool traces; streamed RAG responses.
- Shared settings, provider adapters, error types, logging, uv lockfile, deterministic tests, and setup documentation.

## Verification performed

- Python interpreter: 3.11.9.
- `uv lock --check`: passed; 140 packages resolved.
- `uv run pytest -q`: 16 passed, 1 skipped. The skipped test is the explicitly opt-in OpenAI live smoke test.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- Streamlit launched locally and `/_stcore/health` returned HTTP 200.
- No paid LLM or embedding API calls were made during verification.

The ChromaDB 0.5 adapter emits Pydantic deprecation warnings under current Pydantic 2.x; these are dependency warnings and all persistence tests pass.

## Known limits

- Embeddings currently require OpenAI credentials; generation can select OpenAI or Gemini.
- Agent native tool calling currently requires OpenAI.
- Scanned PDFs need OCR, which is not included.
- Conversation history lasts for the Streamlit session; durable chat storage is not configured.
- Similarity thresholds are retrieval heuristics and do not establish factual correctness.
