"""Streamlit front end; task logic remains in service modules."""

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import streamlit as st

from genai_engineering.core.config import get_settings
from genai_engineering.core.logging import configure_logging
from genai_engineering.core.providers import create_provider
from genai_engineering.task1_structured_extraction.extractor import Extractor
from genai_engineering.task3_agent.agent import ToolAgent
from genai_engineering.task4_chatbot.memory import append_message, new_conversation
from genai_engineering.task4_chatbot.persistence import (
    ConversationRepository,
    stable_owner_id,
)

configure_logging()
st.set_page_config(
    page_title="GenAI Engineering Workbench", page_icon="🧩", layout="wide"
)
st.title("GenAI Engineering Workbench")
if "conversation" not in st.session_state:
    st.session_state.conversation = new_conversation()
if "history" not in st.session_state:
    st.session_state.history = []
settings = get_settings()
if "chat_owner_id" not in st.session_state:
    upstream_identity = (
        st.context.headers.get(settings.chat_owner_header)
        if settings.chat_owner_header
        else None
    )
    if upstream_identity and settings.chat_session_secret:
        st.session_state.chat_owner_id = stable_owner_id(
            upstream_identity, settings.chat_session_secret
        )
    else:
        st.session_state.chat_owner_id = str(uuid4())
repository = (
    ConversationRepository(settings.chat_database_path)
    if settings.chat_persistence_enabled
    else None
)
if repository and st.session_state.conversation["id"] not in {
    item["id"] for item in repository.list(st.session_state.chat_owner_id)
}:
    repository.create(
        st.session_state.chat_owner_id, st.session_state.conversation["id"]
    )


def save_message(role: str, content: str, metadata: dict | None = None) -> None:
    append_message(st.session_state.conversation, role, content)
    if repository:
        repository.append(
            st.session_state.chat_owner_id,
            st.session_state.conversation["id"],
            role,
            content,
            metadata,
        )


with st.sidebar:
    st.caption(f"Conversation {st.session_state.conversation['id'][:8]}")
    if repository:
        saved = repository.list(st.session_state.chat_owner_id)
        saved_by_id = {item["id"]: item for item in saved}
        if saved:
            current_id = st.session_state.conversation["id"]
            selected_id = st.selectbox(
                "Saved conversations",
                options=list(saved_by_id),
                index=(
                    list(saved_by_id).index(current_id)
                    if current_id in saved_by_id
                    else 0
                ),
                format_func=lambda item_id: saved_by_id[item_id]["title"],
            )
            if selected_id != current_id:
                st.session_state.conversation = repository.load(
                    st.session_state.chat_owner_id, selected_id
                )
    if st.button("New conversation"):
        st.session_state.conversation = new_conversation()
        st.session_state.history = []
        if repository:
            repository.create(
                st.session_state.chat_owner_id, st.session_state.conversation["id"]
            )
        st.rerun()
    if repository and st.button("Delete current conversation"):
        repository.delete(
            st.session_state.chat_owner_id, st.session_state.conversation["id"]
        )
        st.session_state.conversation = new_conversation()
        repository.create(
            st.session_state.chat_owner_id, st.session_state.conversation["id"]
        )
        st.rerun()
    mode = st.radio(
        "Mode", ["Structured extraction", "Document Q&A", "Autonomous agent"]
    )
    st.caption("Provider credentials are read from the local .env file.")

for item in st.session_state.conversation["messages"]:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

if mode == "Structured extraction":
    kind = st.selectbox("Document type", ["invoice", "resume"])
    upload = st.file_uploader("Upload text file", type=["txt", "md"])
    text = st.text_area("Or paste document text", height=180)
    if st.button("Extract", type="primary"):
        content = text or (
            upload.getvalue().decode("utf-8", errors="replace") if upload else ""
        )
        try:
            result = Extractor(create_provider()).extract(content, kind)
            serialized = Extractor.to_json(result)
            st.json(json.loads(serialized))
            st.download_button(
                "Download JSON",
                serialized,
                file_name=f"{kind}.json",
                mime="application/json",
            )
        except Exception as exc:  # noqa: BLE001 - render expected runtime/API failures in the UI
            st.error(str(exc))
elif mode == "Document Q&A":
    st.info(
        "Upload documents to the configured persistent collection. PDF page "
        "citations are kept with each chunk. OCR requires Tesseract installation."
    )
    uploads = st.file_uploader(
        "PDF, TXT, PNG, or JPEG documents",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if st.button("Ingest documents") and uploads:
        try:
            from genai_engineering.task2_rag.service import RAGService

            service = RAGService(create_provider())
            total = 0
            upload_temp_root = Path(settings.chroma_path).parent
            upload_temp_root.mkdir(parents=True, exist_ok=True)
            for upload in uploads:
                with tempfile.TemporaryDirectory(dir=upload_temp_root) as temp_dir:
                    temp_path = Path(temp_dir) / Path(upload.name).name
                    temp_path.write_bytes(upload.getvalue())
                    total += service.ingest(str(temp_path))
            st.success(f"Added {total} new chunks.")
        except Exception as exc:  # noqa: BLE001 - ingestion failures should stay recoverable
            st.error(str(exc))
    question = st.chat_input("Ask about your indexed documents")
    if question:
        st.chat_message("user").write(question)
        message_metadata = {"mode": "rag", "provider": settings.llm_provider}
        save_message("user", question, message_metadata)
        try:
            from genai_engineering.task2_rag.service import RAGService

            service = RAGService(create_provider())
            token_stream, sources = service.answer_stream(question)
            with st.chat_message("assistant"):
                answer = st.write_stream(token_stream)
                for source in sources:
                    st.caption(f"{source['source']} · page {source['page']}")
            save_message("assistant", str(answer), message_metadata)
        except Exception as exc:  # noqa: BLE001 - keep provider/retrieval errors visible
            st.error(str(exc))
elif mode == "Autonomous agent":
    prompt = st.chat_input(
        "Describe a task that may need web search, math, or local files"
    )
    if prompt:
        st.chat_message("user").write(prompt)
        message_metadata = {"mode": "agent", "provider": settings.llm_provider}
        save_message("user", prompt, message_metadata)
        try:
            status = st.status("Agent is working", expanded=True)
            answer_area = st.empty()
            answer_so_far = [""]

            def handle_agent_event(event: dict) -> None:
                if event["type"] == "tool_start":
                    status.write(f"Running {event['tool']}…")
                elif event["type"] == "tool_result":
                    status.write(
                        f"{event['tool']}: {event['status']} "
                        f"({event['duration_ms']:.0f} ms)"
                    )
                elif event["type"] == "token":
                    answer_so_far[0] += event["text"]
                    answer_area.markdown(answer_so_far[0] + "▌")

            result = ToolAgent(create_provider()).run(
                prompt, on_event=handle_agent_event
            )
            status.update(label="Agent finished", state="complete", expanded=False)
            with st.chat_message("assistant"):
                for entry in result.trace:
                    with st.expander(f"{entry.tool}: {entry.action}"):
                        st.json(entry.tool_input)
                        st.code(entry.observation)
                        st.caption(f"Next action: {entry.next_action}")
                st.markdown(result.answer)
            save_message("assistant", result.answer, message_metadata)
        except Exception as exc:  # noqa: BLE001 - keep tool/provider failures visible
            st.error(str(exc))
