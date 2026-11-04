import pytest

from genai_engineering.task4_chatbot.persistence import (
    ConversationRepository,
    stable_owner_id,
)


def test_repository_isolates_conversations_by_owner_and_round_trips_messages(tmp_path):
    repository = ConversationRepository(str(tmp_path / "chat.sqlite3"))
    repository.create("owner-a", "conversation-1")
    repository.create("owner-b", "conversation-2")
    repository.append("owner-a", "conversation-1", "user", "Find the report")
    repository.append(
        "owner-a", "conversation-1", "assistant", "Here is the report", {"mode": "rag"}
    )

    assert [item["id"] for item in repository.list("owner-a")] == ["conversation-1"]
    assert repository.load("owner-b", "conversation-1") is None
    loaded = repository.load("owner-a", "conversation-1")
    assert loaded["messages"][1]["metadata"] == {"mode": "rag"}
    assert repository.delete("owner-a", "conversation-1")
    assert repository.load("owner-a", "conversation-1") is None


def test_repository_refuses_cross_owner_writes(tmp_path):
    repository = ConversationRepository(str(tmp_path / "chat.sqlite3"))
    repository.create("owner-a", "conversation-1")

    with pytest.raises(KeyError):
        repository.append("owner-b", "conversation-1", "user", "No access")


def test_stable_owner_pseudonym_is_secret_scoped():
    first = stable_owner_id("user-123", "secret-a")
    assert stable_owner_id("user-123", "secret-a") == first
    assert stable_owner_id("user-456", "secret-a") != first
    assert stable_owner_id("user-123", "secret-b") != first


def test_conversations_survive_repository_recreation(tmp_path):
    path = str(tmp_path / "chat.sqlite3")
    owner = stable_owner_id("user-123", "deployment-secret")
    first_process = ConversationRepository(path)
    first_process.create(owner, "conversation-1")
    first_process.append(owner, "conversation-1", "user", "Persist this")

    restarted_process = ConversationRepository(path)

    assert (
        restarted_process.load(owner, "conversation-1")["messages"][0]["content"]
        == "Persist this"
    )
