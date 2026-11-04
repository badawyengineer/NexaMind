# Deployment and operations

## Local deployment

Install Python 3.11 and uv, copy `.env.example` to `.env`, set the provider key,
then run `uv sync --locked` and `uv run streamlit run app/streamlit_app.py`.
OCR is available when Tesseract and the requested language pack are installed.

## Streamlit-hosted deployment

For a self-managed Streamlit host, install the locked environment, set provider
credentials through the host's secret manager, and start
`uv run streamlit run app/streamlit_app.py`. Mount `CHROMA_PATH` and, when enabled,
`CHAT_DATABASE_PATH` on durable storage. Configure an authenticated TLS proxy in
front of the app. Do not use an ephemeral filesystem when documents or saved chats
must survive a host restart. Managed Streamlit services differ in filesystem
durability and secret handling; verify those settings in the chosen service before
deploying.

## Docker Compose

Build and start the app with `docker compose up --build -d`. Streamlit listens on
port 8501 and the container health endpoint is `/_stcore/health`. Set only the
provider credentials needed by the selected model in `.env`; Compose does not
copy `.env` into the image. Chroma files and the opt-in SQLite chat database are
stored in the `nexamind-data` named volume at `/app/data`.

Create a backup before upgrades:

```powershell
docker compose stop
$volume = docker volume ls --filter label=com.docker.compose.volume=nexamind-data --format '{{.Name}}' |
  Select-Object -First 1
docker run --rm -v "${volume}:/data" -v "${PWD}:/backup" alpine `
  tar -czf /backup/nexamind-data-backup.tar.gz -C /data .
docker compose start
```

Restore by stopping the app, extracting the archive into the named volume, then
starting the app again:

```powershell
docker compose stop
$volume = docker volume ls --filter label=com.docker.compose.volume=nexamind-data --format '{{.Name}}' |
  Select-Object -First 1
docker run --rm -v "${volume}:/data" -v "${PWD}:/backup" alpine `
  sh -c "tar -xzf /backup/nexamind-data-backup.tar.gz -C /data"
docker compose start
```

The volume name is prefixed with the Compose project name. Keep backup archives
outside the repository and protect them like the original documents. Avoid
`docker compose down --volumes` unless you intend to delete all persisted data.

## Security review

- Provider credentials come from environment configuration and are not written to
  conversation messages or logs. Keep `.env` out of version control and rotate
  keys if exposed.
- Uploaded paths are reduced to their filename before writing temporary files.
  Ingestion validates file signatures for PDF/PNG/JPEG, applies byte, PDF page,
  image-pixel, and configured OCR bounds, and rejects unsupported formats.
- Agent file reads resolve paths under `ALLOWED_FILE_ROOT`, validate extensions
  and size, and return bounded text. Calculator input is parsed by a restricted
  AST; it does not use `eval`.
- Retrieved documents, OCR output, and web results are treated as untrusted prompt
  content. The model receives an instruction to ignore embedded instructions.
  Tool arguments are validated before execution and each agent run has a bounded
  total-call budget.
- SQLite conversation persistence is off by default. When enabled, rows are
  scoped to a random Streamlit session owner ID and every read/write/delete query
  checks that owner. To restore the same conversation list after a Streamlit
  restart, configure `CHAT_OWNER_HEADER` and `CHAT_SESSION_SECRET` behind a trusted
  authentication proxy that replaces that header; the app stores only an HMAC
  pseudonym of the identity. Without that proxy configuration, a new Streamlit
  session receives a new owner ID. The local volume and backups may still contain
  personal chat content, so limit filesystem access and retention.
- Observability records provider/model identifiers, operation names, status,
  duration, and token counts when available. Prompts, retrieved text, API keys,
  and tool arguments are not included in structured events.
- The container runs as an unprivileged user, enables Streamlit XSRF protection,
  and keeps persistent state on a mounted volume. Place it behind an authenticated
  TLS reverse proxy before exposing it outside a trusted network.
- GitHub Actions uses read-only repository permissions, does not receive model
  secrets, installs the locked dependency set, and runs a dependency vulnerability
  audit. The current `chromadb==0.5.23` lock reports three Chroma advisories:
  PYSEC-2026-3813 (CVE-2026-45830, cross-tenant access), PYSEC-2026-3814
  (CVE-2026-45833, collection-update code execution when loading untrusted model
  repositories with `trust_remote_code`), and PYSEC-2026-3815 (CVE-2026-45831,
  resource-scope checks missing from `SimpleRBACAuthorizationProvider`). The CI
  audit suppresses only those exact IDs because this application uses Chroma's
  embedded persistent client, does not expose Chroma's FastAPI server, does not
  select `SimpleRBACAuthorizationProvider`, and does not load user-selected model
  repositories with `trust_remote_code`. Chroma's latest published 1.5.9 is also
  listed as affected and no patched stable release is available in the current
  advisory data, so upgrading across the major API line is not a verified fix.
  Recheck these advisories before any server-mode deployment or dependency
  upgrade; all other audit findings fail CI.

The application does not implement account authentication, per-user tenancy,
remote database encryption, or automatic data-retention deletion. Do not expose
the local deployment directly to an untrusted public network.
