from genai_engineering.task2_rag.vector_store import ChromaStore


def test_chroma_persistence_api_add_search_and_delete(tmp_path):
    store = ChromaStore(str(tmp_path / "vectors"), "test_docs")
    chunk = {
        "id": "hash:1:0",
        "text": "The answer is forty two.",
        "metadata": {"source": "guide.txt", "source_id": "hash", "page": 1},
    }
    assert store.add([chunk], [[1.0, 0.0]]) == 1
    assert store.add([chunk], [[1.0, 0.0]]) == 0
    hits = store.search([1.0, 0.0], top_k=1)
    assert hits[0]["text"] == chunk["text"]
    assert hits[0]["metadata"]["page"] == 1
    store.delete_source("hash")
    assert store.search([1.0, 0.0], top_k=1) == []
