from langchain_text_splitters import RecursiveCharacterTextSplitter

from genai_engineering.task2_rag.ingestion import Page


def split_pages(
    pages: list[Page],
    *,
    chunk_size: int = 900,
    overlap: int = 120,
    min_length: int = 40,
) -> list[dict]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True
    )
    chunks = []
    for page in pages:
        for item in splitter.create_documents([page.text]):
            content = item.page_content.strip()
            if len(content) < min_length:
                continue
            idx = item.metadata["start_index"]
            chunk_id = f"{page.source_id}:{page.page}:{idx}"
            chunks.append(
                {
                    "text": content,
                    "metadata": {
                        "source": page.source,
                        "source_id": page.source_id,
                        "page": page.page,
                        "extraction_method": page.extraction_method,
                        "start_index": idx,
                        "chunk_id": chunk_id,
                    },
                    "id": chunk_id,
                }
            )
    return chunks
