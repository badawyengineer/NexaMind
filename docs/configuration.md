# Configuration reference

Copy `.env.example` to `.env`. Environment variables override values in the file.
The settings object validates numeric ranges and ignores unknown variables.

| Variables | Purpose | Defaults / notes |
|---|---|---|
| `LLM_PROVIDER`, `OPENAI_API_KEY`, `GEMINI_API_KEY` | Text and agent provider selection/credentials | `openai`; provide only the selected provider's key. |
| `OPENAI_MODEL`, `GEMINI_MODEL` | Text-generation model names | `gpt-4o-mini`, `gemini-2.0-flash`. |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | Independent RAG embedding selection | `openai`, `text-embedding-3-small`; supports `gemini` and `sentence_transformers`. |
| `EMBEDDING_DIMENSIONS`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_REVISION`, `EMBEDDING_DEVICE` | Embedding output/configuration | Dimension inferred where known; batch 128; revision `default`; device `cpu`. |
| `CHROMA_PATH`, `CHROMA_COLLECTION` | Persistent vector storage | `data/vectorstore`, `genai_documents`. Use a fresh collection on model changes. |
| `ALLOWED_FILE_ROOT` | Agent reader's confined directory | `data/samples`. |
| `MAX_FILE_BYTES`, `MAX_DOCUMENT_PAGES`, `MAX_IMAGE_PIXELS` | Ingestion/resource bounds | 5,000,000 bytes, 100 pages, 40,000,000 pixels. |
| `OCR_ENABLED`, `OCR_LANGUAGE`, `OCR_FALLBACK_THRESHOLD`, `OCR_DPI`, `TESSERACT_CMD` | OCR behavior | Disabled by default; language `eng`; threshold 20 characters; 200 DPI. Tesseract and language data are external. |
| `CHAT_PERSISTENCE_ENABLED`, `CHAT_DATABASE_PATH` | Optional SQLite chat history | Disabled; `data/conversations.sqlite3`. |
| `CHAT_OWNER_HEADER`, `CHAT_SESSION_SECRET` | Stable owner identity across app restarts | Optional. Only use an identity header replaced by a trusted authentication proxy. |
| `TEMPERATURE`, `REQUEST_TIMEOUT`, `MAX_RETRIES` | Provider behavior | 0, 45 seconds, 2 retries. |

Do not commit `.env`, provider credentials, `CHAT_SESSION_SECRET`, SQLite files, or
vector-store data. See [deployment.md](deployment.md) for authenticated deployment,
volumes, backups, and the Chroma advisory exception.
