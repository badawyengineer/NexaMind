import pytest

from genai_engineering.task4_chatbot.memory import append_message, new_conversation


def test_conversation_isolated_and_typed():
    a, b = new_conversation(), new_conversation()
    append_message(a, "user", "hi")
    assert a["id"] != b["id"] and b["messages"] == []
    with pytest.raises(ValueError):
        append_message(a, "system", "no")
