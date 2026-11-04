# NexaMind completion audit

Audit date: 2026-11-04
Repository: `F:\genai-engineering`
Baseline commit: `8e13c32` (`main`, clean worktree)

## Baseline verification

- Python: 3.11.9; uv-managed environment and lockfile present.
- `uv run pytest -q`: 16 passed, 1 opt-in live test skipped. Chroma emits Pydantic v2 deprecation warnings.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- `uv lock --check`: passed, 140 packages resolved.
- `.env` is absent; `.env.example` has blank credentials; `.gitignore` excludes `.env`, `.venv`, logs, and the local vector store.

## Milestone findings

| Milestone | Current Status | Missing Work | Files Affected | Priority |
|---|---|---|---|---|
| M6 | **Implemented and verified**: provider protocol/factory, OpenAI/Gemini adapters, optional Sentence Transformers adapter, batch controls, and Chroma fingerprints with legacy/mismatch rejection. Focused tests and the full suite pass. | Optional Sentence Transformers runtime/model download not exercised due the large Torch/model footprint; deterministic adapter coverage is in place. | `core/embeddings.py`, `core/config.py`, `task2_rag/vector_store.py`, `task2_rag/service.py`, `scripts/migrate_embeddings.py`, tests/docs | Critical |
| M7 | **Implemented and focused-tested**: PyMuPDF page extraction, TXT and PNG/JPEG support, optional Tesseract OCR fallback, extraction-method metadata, MIME/byte/page/pixel bounds, and clear OCR errors. | Native Tesseract executable and language data require separate installation; end-to-end OCR not exercised in this environment. | `task2_rag/ingestion.py`, `task2_rag/service.py`, `core/config.py`, `pyproject.toml`, `.env.example`, ingestion tests/docs | High |
| M8 | **Implemented and verified**: provider-neutral tool-call records/adapters for OpenAI and Gemini; shared schemas derive from validated Pydantic arguments; total call budget and status/duration traces. Focused and full tests pass. | Live provider calls require credentials and may incur charges; adapter behavior is covered with fakes. | `task3_agent/provider.py`, `task3_agent/agent.py`, tests/docs | High |
| M9 | **Implemented and tested** deterministic chunk rank/context/answer proxy/abstention/citation checks, eight curated cases including untrusted and conflicting fixtures, a chunk/top-k/threshold matrix runner, saved-run JSON/Markdown reports, and sanitized structured OpenTelemetry spans with latency/token usage. | Semantic faithfulness remains a human review task; live evaluation calls are explicit and can incur provider charges. | `task2_rag/evaluation.py`, `data/evaluation/`, `core/observability.py`, provider/RAG services, tests, evaluation scripts/docs | Critical |
| M10 | **Implemented and tested** opt-in owner-scoped SQLite storage, conversation list/load/delete UI, and provider-streamed agent tokens plus tool progress/status. RAG already streamed. | Stable retrieval across Streamlit sessions requires configuring a trusted auth-proxy owner header and session secret; otherwise each session gets a new owner ID. | `task4_chatbot/persistence.py`, `core/config.py`, `task3_agent/agent.py`, provider adapters, Streamlit app, tests/docs | Medium-high |
| M11 | **Implemented** locked CI/lint/tests, pip-audit job, Docker/Compose persistent volume and healthcheck, unprivileged container, upload limits, deployment/backup and security review docs. Compose config and pip-audit passed with three narrowly documented Chroma advisory exceptions. | Image build/runtime smoke was not run because the Docker image store is on the full system drive. | `.github/workflows/`, Docker files, `.streamlit/`, ingestion/config, docs | High |

## Baseline cross-cutting observations

The following records describe the repository at the baseline commit above. The M6–M11 status table reflects the completed implementation.

- The business logic is already separated into extraction, RAG, agent, and UI packages; these boundaries can remain.
- `LLMProvider` currently describes only text generation and streaming. The agent bypasses that protocol and checks the concrete `OpenAIProvider` class.
- `RAGService` injects embeddings and a store, which is a useful seam for provider and compatibility work.
- Chroma stores only `hnsw:space=cosine`; existing collections cannot identify which embedding model created their vectors.
- PDF ingestion uses `pypdf` and does not support images or OCR. `load_document` hashes the whole file after enforcing a byte limit.
- Tool failures become observations, but traces have no status or duration and one model turn can return multiple tool calls beyond the nominal iteration count.
- Conversation messages have no persistent identity beyond an in-memory UUID and are not stored outside Streamlit session state.
- The current docs correctly state that Gemini agent tools, local embeddings, OCR, and durable chat storage are absent.

## Remaining risks and constraints

- Sentence Transformers adds a sizeable local model/runtime dependency and downloaded model weights. Keep it optional so OpenAI-only installs remain light; validate disk/cache availability before resolving it.
- Existing Chroma collections lack an embedding fingerprint. Treat them as legacy/unknown and require explicit migration or a new collection instead of guessing compatibility.
- Tesseract is an external executable and language data are installed separately from Python packages. OCR must be opt-in and report a clear setup error.
- Gemini function calling must be mapped through the installed official `google-genai` SDK and tested with fake responses; do not couple tool execution to Gemini types.
- SQLite content may contain personal data. Persistence must be opt-in and every lookup/list/delete must be scoped to a random session owner key.
- CI and Docker must never receive API secrets for normal checks. Native execution remains supported.

## Original implementation sequence

This was the implementation plan at the start of the work; the status table above records its completion.

1. **M6 embeddings** — establish a small embedding protocol and factory; add OpenAI and optional local Sentence Transformers adapters; persist and validate collection fingerprints; reject legacy/mismatched collections with a migration path; add batch-size configuration and deterministic fakes.
2. **M7 OCR/document processing** — replace the single loader internals with page/image processing while keeping the `Page`/chunking boundary; text extraction first, OCR fallback; preserve page and method metadata; add byte/page/image limits and mocked OCR tests.
3. **M8 provider-neutral agent** — define shared tool contracts, provider tool-call request/response records, OpenAI and Gemini adapters, then keep one execution loop with Pydantic validation, a total-call budget, status/duration traces, and fake adapter tests.
4. **M9 evaluation/observability** — build deterministic retrieval metrics and a richer fixture set first; add answer/citation/abstention checks and JSON/Markdown reports; instrument providers, embeddings, retrieval, and tool calls with sanitized structured events and timings.
5. **M10 SQLite and streaming** — implement an optional stdlib SQLite repository keyed by session ID, wire history/list/new/delete UI, then emit agent progress/final events through provider adapters while retaining the existing RAG stream.
6. **M11 CI/deployment/security** — add GitHub Actions with locked uv, lint/format/tests and dependency scanning; add Docker/Compose with persistent data and healthcheck; tighten MIME/page limits and complete the security/deployment docs.

Each milestone depends on the interfaces created immediately before it. Keep the existing extraction schemas, chunker, Chroma backend, calculator, file reader, and Streamlit service calls unless tests demonstrate a required change. After each milestone, run its focused tests plus the full unit suite and Ruff, then update the engineering report.
