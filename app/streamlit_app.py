"""Streamlit front end; task logic remains in service modules."""

import json
import tempfile
from pathlib import Path

import streamlit as st

from genai_engineering.core.logging import configure_logging
from genai_engineering.core.providers import create_provider
from genai_engineering.task1_structured_extraction.extractor import Extractor
from genai_engineering.task3_agent.agent import ToolAgent
from genai_engineering.task4_chatbot.memory import append_message, new_conversation

configure_logging()
st.set_page_config(
    page_title="GenAI Engineering Workbench", page_icon="🧩", layout="wide"
)
st.title("GenAI Engineering Workbench")
if "conversation" not in st.session_state:
    st.session_state.conversation = new_conversation()
if "history" not in st.session_state:
    st.session_state.history = []
with st.sidebar:
    st.caption(f"Conversation {st.session_state.conversation['id'][:8]}")
    if st.button("New conversation"):
        st.session_state.conversation = new_conversation()
        st.session_state.history = []
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
        "Upload documents to the configured persistent collection. "
        "PDF page citations are kept with each chunk."
    )
    uploads = st.file_uploader(
        "PDF or TXT documents", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Ingest documents") and uploads:
        try:
            from genai_engineering.task2_rag.service import RAGService

            service = RAGService(create_provider())
            total = 0
            for upload in uploads:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir) / Path(upload.name).name
                    temp_path.write_bytes(upload.getvalue())
                    total += service.ingest(str(temp_path))
            st.success(f"Added {total} new chunks.")
        except Exception as exc:  # noqa: BLE001 - ingestion failures should stay recoverable
            st.error(str(exc))
    question = st.chat_input("Ask about your indexed documents")
    if question:
        st.chat_message("user").write(question)
        append_message(st.session_state.conversation, "user", question)
        try:
            from genai_engineering.task2_rag.service import RAGService

            service = RAGService(create_provider())
            token_stream, sources = service.answer_stream(question)
            with st.chat_message("assistant"):
                answer = st.write_stream(token_stream)
                for source in sources:
                    st.caption(f"{source['source']} · page {source['page']}")
            append_message(st.session_state.conversation, "assistant", str(answer))
        except Exception as exc:  # noqa: BLE001 - keep provider/retrieval errors visible
            st.error(str(exc))
elif mode == "Autonomous agent":
    prompt = st.chat_input(
        "Describe a task that may need web search, math, or local files"
    )
    if prompt:
        st.chat_message("user").write(prompt)
        append_message(st.session_state.conversation, "user", prompt)
        try:
            result = ToolAgent(create_provider()).run(prompt)
            with st.chat_message("assistant"):
                for entry in result.trace:
                    with st.expander(f"{entry.tool}: {entry.action}"):
                        st.json(entry.tool_input)
                        st.code(entry.observation)
                        st.caption(f"Next action: {entry.next_action}")
                st.markdown(result.answer)
            append_message(st.session_state.conversation, "assistant", result.answer)
        except Exception as exc:  # noqa: BLE001 - keep tool/provider failures visible
            st.error(str(exc))
