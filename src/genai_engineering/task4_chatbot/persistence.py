"""Opt-in SQLite conversation storage scoped to a random browser session ID."""

import hashlib
import hmac
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator


def stable_owner_id(identity: str, secret: str) -> str:
    """Pseudonymize a trusted upstream identity for cross-restart chat access."""
    return hmac.new(
        secret.encode("utf-8"), identity.encode("utf-8"), hashlib.sha256
    ).hexdigest()


class ConversationRepository:
    def __init__(self, database_path: str):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT 'New conversation'
                );
                CREATE INDEX IF NOT EXISTS conversations_owner_created
                    ON conversations(owner_id, created_at);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create(self, owner_id: str, conversation_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id, owner_id, created_at) VALUES (?, ?, ?)",
                (conversation_id, owner_id, datetime.now(UTC).isoformat()),
            )

    def list(self, owner_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at FROM conversations "
                "WHERE owner_id=? ORDER BY created_at DESC",
                (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def load(self, owner_id: str, conversation_id: str) -> dict | None:
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT id, title FROM conversations WHERE id=? AND owner_id=?",
                (conversation_id, owner_id),
            ).fetchone()
            if conversation is None:
                return None
            messages = connection.execute(
                "SELECT role, content, metadata_json FROM messages "
                "WHERE conversation_id=? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return {
            "id": conversation["id"],
            "messages": [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata_json"]),
                }
                for row in messages
            ],
        }

    def append(
        self,
        owner_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Invalid message role")
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id=? AND owner_id=?",
                (conversation_id, owner_id),
            ).fetchone()
            if exists is None:
                raise KeyError("Conversation does not belong to this session")
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at, "
                "metadata_json) VALUES (?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    role,
                    content,
                    datetime.now(UTC).isoformat(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            if role == "user":
                title = content.strip().replace("\n", " ")[:80] or "New conversation"
                connection.execute(
                    "UPDATE conversations SET title=? "
                    "WHERE id=? AND owner_id=? AND title='New conversation'",
                    (title, conversation_id, owner_id),
                )

    def delete(self, owner_id: str, conversation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id=? AND owner_id=?",
                (conversation_id, owner_id),
            )
        return cursor.rowcount == 1
