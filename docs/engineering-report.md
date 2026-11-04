# Engineering report

Date: 2026-11-04

## Delivered

- Task 1: invoice and resume Pydantic v2 schemas, parameterized prompts, OpenAI/Gemini text adapters, JSON validation, bounded repair, serialization, and sample input/output files.
- Task 2: bounded TXT/PDF/image ingestion with optional OCR, page-level extraction metadata, recursive chunking, OpenAI/Gemini/optional local batched embeddings, fingerprinted persistent Chroma storage, legacy migration support, duplicate avoidance, filtered similarity retrieval, grounded generation, citations, and deterministic evaluation helpers.
- Task 3: provider-neutral OpenAI/Gemini function-call orchestration over web search, AST calculator, and confined local reader, with Pydantic argument schemas, total-call limits, streamed final tokens, observations, status, and duration traces.
- Task 4: Streamlit extraction, RAG, and agent modes; session-scoped conversations; optional owner-isolated SQLite persistence; uploads; source citations; tool traces; streamed RAG and agent responses.
- Shared settings, provider adapters, sanitized OpenTelemetry spans/events, error types, uv lockfile, deterministic tests, CI, Docker/Compose, and operations/security documentation.

## Verification performed

- Python interpreter: 3.11.9.
- `uv lock --check`: passed; 177 packages resolved.
- `uv run pytest -q`: 44 passed, 2 skipped; 34 dependency deprecation warnings. The opt-in OpenAI and Tesseract tests are skipped by default.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- `docker compose config`: passed. Docker image build was not run because the Docker image store is on the system drive with no free space.
- `pip-audit --strict --no-deps --disable-pip --vulnerability-service osv`: passed with six advisory records ignored for three documented Chroma IDs. No other known findings were reported.
- No paid LLM or embedding API calls were made during verification. OCR and provider adapters use local fakes in unit tests.

The ChromaDB 0.5 adapter emits Pydantic deprecation warnings under current Pydantic 2.x; these are dependency warnings and all persistence tests pass.

## Known limits

- OpenAI and Gemini embeddings need their API credentials; Sentence Transformers is local after installing the optional extra and downloading a model.
- The Sentence Transformers extra and real local model inference were not installed; mocked adapter tests verify the protocol.
- OCR requires the external Tesseract executable and separately installed language data.
- Semantic answer faithfulness is not measured automatically; lexical checks are proxies that require human review.
- Docker image build/runtime was not executed due to the full system drive.
- The audit exception applies only to the embedded local Chroma mode documented in `docs/deployment.md`; do not expose a Chroma server with this lockfile.
- Similarity thresholds are retrieval heuristics and do not establish factual correctness.
