# NexaMind final technical report

Date: 2026-11-04

## Delivery summary

The existing extraction, RAG, agent, and Streamlit application boundaries remain
in place. M6–M11 now add provider-independent embeddings, fingerprinted Chroma
collections and re-embedding migration, bounded OCR/document ingestion,
provider-neutral OpenAI/Gemini tool calling, reproducible evaluation, sanitized
observability, opt-in SQLite chat persistence, streamed agent output, CI, and
container deployment files.

See [architecture.md](architecture.md) for component/data flows,
[testing-report.md](testing-report.md) for executed checks, and
[security-review.md](security-review.md) for security findings and limits.

## Verification state

- Unit suite, lint, format, lockfile validation, evaluation CLI help, and Docker
  Compose configuration were checked locally.
- Final unit suite: 44 passed, 2 skipped; the skips are opt-in live OpenAI and
  Tesseract checks.
- The pip-audit scan passed after suppressing only three documented Chroma advisory
  IDs; no other findings were reported.
- The container build and HTTP health check are configured as a CI job, but were
  not run in this Windows workspace because Docker stores image layers on the full
  system drive.
- OpenAI/Gemini live calls, a real Sentence Transformers model download, and
  Tesseract OCR are opt-in and were not used in normal verification.

## Remaining operational work

- Install a local Sentence Transformers model or configure an embedding API before
  ingesting documents.
- Install Tesseract plus language data for OCR. The opt-in real OCR test is
  documented in [evaluation.md](evaluation.md).
- Configure `CHAT_OWNER_HEADER` and `CHAT_SESSION_SECRET` behind a trusted auth
  proxy if saved conversations must be visible to the same identity after a
  Streamlit process restart. Persistence remains off by default.
- Review the Chroma advisory exception before changing from local embedded mode,
  exposing a Chroma server, or upgrading Chroma.

This report does not claim that the app was deployed or that the Docker image was
run successfully in this environment.
