"""多轮会话存储（v1.1 W1）。

SQLite 单文件存储：
- WAL 模式 + busy_timeout=3000ms，适配 API 多线程读写；
- 每次操作使用独立连接，避免跨线程共享连接；
- 消息按自增 id 排序，时间戳统一为 UTC ISO8601。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import get_settings

DEFAULT_SESSION_TITLE = "新会话"
BUSY_TIMEOUT_MS = 3000
MAX_TITLE_CHARS = 200
SESSION_COLUMNS = "id, title, created_at, updated_at"
MESSAGE_COLUMNS = "id, session_id, role, content, meta_json, created_at"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _parse_meta(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _session_row(row: sqlite3.Row, message_count: int = 0) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "message_count": int(message_count),
    }


def _message_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "session_id": row["session_id"],
        "role": row["role"],
        "content": row["content"],
        "meta": _parse_meta(row["meta_json"]),
        "created_at": row["created_at"],
    }


class SessionStore:
    """会话与消息的 SQLite 存取；可注入路径，默认取 settings.session_store_path。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else get_settings().session_store_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ---------- connection ----------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=BUSY_TIMEOUT_MS / 1000)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.executescript(SCHEMA)

    # ---------- sessions ----------
    def create_session(self, title: str | None = None) -> dict[str, Any]:
        session_id = _new_id("s")
        now = _now()
        safe_title = (title or "").strip()[:MAX_TITLE_CHARS] or DEFAULT_SESSION_TITLE
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, safe_title, now, now),
            )
        return {
            "id": session_id,
            "title": safe_title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """返回会话详情（含 messages）；不存在返回 None。"""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {SESSION_COLUMNS} FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            messages = conn.execute(
                f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        payload = _session_row(row, len(messages))
        payload["messages"] = [_message_row(m) for m in messages]
        return payload

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """按最近更新时间倒序返回会话摘要（不含 messages）。"""
        safe_limit = max(1, int(limit))
        safe_offset = max(0, int(offset))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count
                FROM sessions s
                ORDER BY s.updated_at DESC, s.id DESC
                LIMIT ? OFFSET ?
                """,
                (safe_limit, safe_offset),
            ).fetchall()
        return [_session_row(row, row["message_count"]) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        """删除会话（消息级联删除）；返回是否命中。"""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    # ---------- messages ----------
    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """追加消息并刷新会话 updated_at；会话不存在时抛 KeyError。"""
        now = _now()
        meta_json = json.dumps(meta or {}, ensure_ascii=False, default=str)
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if exists is None:
                raise KeyError(f"session not found: {session_id}")
            cursor = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, meta_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, meta_json, now),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        return {
            "id": int(cursor.lastrowid or 0),
            "session_id": session_id,
            "role": role,
            "content": content,
            "meta": meta or {},
            "created_at": now,
        }

    def get_history(
        self,
        session_id: str,
        max_turns: int = 6,
        max_chars: int = 4000,
    ) -> list[dict[str, Any]]:
        """按时间正序返回最近 max_turns 轮（每轮 user+assistant）且不超过 max_chars 的历史。

        预算优先保留最新消息：超预算时从最旧的一条开始丢弃，至少保留最新一条。
        """
        safe_turns = max(0, int(max_turns))
        safe_chars = max(0, int(max_chars))
        if safe_turns == 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, meta_json, created_at
                FROM messages WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, safe_turns * 2),
            ).fetchall()
        items = [
            {
                "role": row["role"],
                "content": row["content"],
                "meta": _parse_meta(row["meta_json"]),
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]
        total_chars = sum(len(item["content"]) for item in items)
        while len(items) > 1 and total_chars > safe_chars:
            total_chars -= len(items[0]["content"])
            items.pop(0)
        return items
