from uuid import uuid4


def new_conversation() -> dict:
    return {"id": str(uuid4()), "messages": []}


def append_message(conversation: dict, role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError("Invalid message role")
    conversation["messages"].append({"role": role, "content": content})
