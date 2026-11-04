# Provider integration guide

## Text and agent providers

Set `LLM_PROVIDER=openai` with `OPENAI_API_KEY`, or
`LLM_PROVIDER=gemini` with `GEMINI_API_KEY`. Select the generation model using the
matching `OPENAI_MODEL` or `GEMINI_MODEL`. Extraction, RAG answer generation, and
the single agent orchestration loop use the shared text provider boundary. Agent
tool calling maps the same Pydantic-validated tool definitions through the
OpenAI/Gemini adapter; model keys are not shared with the embedding provider.

## Embedding providers

Choose independently with `EMBEDDING_PROVIDER`:

- `openai`: set `OPENAI_API_KEY`; the default model is `text-embedding-3-small`.
- `gemini`: set `GEMINI_API_KEY`; the default model is `gemini-embedding-001`.
- `sentence_transformers`: install `uv sync --extra local-embeddings`, then set
  `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` and optionally
  `EMBEDDING_DEVICE=cpu` or a supported device. The model downloads on first use.

Document batches use `EMBEDDING_BATCH_SIZE`. Each collection records provider,
model, revision, and dimension. Changing those settings against an existing
collection raises a compatibility error. Re-embed into a new collection using
the procedure in [completion-audit.md](completion-audit.md) and the command
`uv run python scripts/migrate_embeddings.py --source-collection old --target-collection new`.

Unit tests use mocks and do not call a provider. Live calls from the app or
evaluation runner may incur API costs. See [evaluation.md](evaluation.md) for
explicit answer-generation controls.
