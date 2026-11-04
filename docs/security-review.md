# Security review

Date: 2026-11-04

## Controls reviewed

- **Secrets:** provider keys and the optional chat HMAC secret come from
  environment settings. Structured events omit prompts, documents, arguments, and
  credential values. `.env` and generated state are excluded from Git and Docker
  build context.
- **Uploads and resource bounds:** ingestion enforces byte, page, and pixel limits;
  checks PDF/PNG/JPEG signatures; reports OCR-disabled scanned pages; and uses
  Tesseract only when configured. Streamlit's upload cap is 20 MB while the RAG
  default is 5 MB per file.
- **Filesystem:** the agent reader resolves paths under `ALLOWED_FILE_ROOT`,
  restricts extensions and size, limits PDF pages and returned text, and rejects
  path traversal. The calculator uses an allowlisted AST evaluator rather than
  `eval`.
- **Prompt injection and tools:** retrieved documents, OCR text, and web results
  are treated as untrusted. Tool schemas are shared and Pydantic-validated, only
  allowlisted operations execute, and each agent run has iteration and total-call
  caps.
- **Vector compatibility:** Chroma collection fingerprints prevent silent model or
  dimension mixing. Migration writes to a new collection and does not delete the
  source.
- **Conversation persistence:** SQLite is opt-in. Every query scopes by owner ID;
  a stable owner across process restarts requires a trusted authentication proxy
  header and an HMAC secret. Do not trust a client-supplied identity header.
- **Deployment:** Docker runs as a non-root user, persists Chroma/SQLite in a
  volume, configures health checks, and enables Streamlit XSRF protection. Public
  deployment requires an authenticated TLS reverse proxy.

## Dependency audit finding

The current lock uses `chromadb==0.5.23`. `pip-audit` reported six records for
three advisory IDs:

- `PYSEC-2026-3813` / CVE-2026-45830: cross-tenant access.
- `PYSEC-2026-3814` / CVE-2026-45833: collection-update code execution when
  untrusted model repositories are loaded with `trust_remote_code`.
- `PYSEC-2026-3815` / CVE-2026-45831: resource-scope checks missing from
  `SimpleRBACAuthorizationProvider`.

The CI audit ignores only these exact IDs. NexaMind uses Chroma's embedded
persistent client; it does not expose the Chroma FastAPI server, configure that
RBAC provider, or load user-selected embedding repositories with
`trust_remote_code`. Published advisory data also flags Chroma 1.5.9 and does not
list a patched stable release, so a major-version upgrade is not a verified fix.
This is a deployment-specific exception, not a claim that Chroma is vulnerability
free. Re-audit before changing storage mode or upgrading Chroma, and do not expose
a Chroma server with this lockfile.

## Residual risks

- The Streamlit app has no built-in authentication or multi-tenant account system.
  Public access must go through a trusted authentication proxy; otherwise keep it
  on a private network.
- Local SQLite/Chroma volumes and backups contain user content. Protect host
  permissions, backup archives, and retention schedules.
- Tesseract output quality and retrieval thresholds do not prove document
  correctness. Review citations and OCR output for sensitive workflows.
- The Docker image and HTTP health check were not run locally due to the full
  Docker image store drive. CI defines the build/smoke job, but its remote result
  remains unverified here.
- Semantic faithfulness is not automatically judged. The evaluation metrics that
  compare tokens/phrases are proxies and require human review.
