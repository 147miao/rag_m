"""SQLite 任务表与状态机（v1.1 W4 异步入库）。

设计约定：
- 每个操作使用独立连接，WAL + busy_timeout=3000ms，适配 API 线程与 worker 线程并发；
- 状态：queued → running → succeeded | failed；queued/running → cancelled；
- 自动重试由 runner 调用 schedule_retry() 写 next_run_at 实现（attempts 保留）；
- 手动 retry() 只允许 failed → queued，并重置 attempts；
- recover() 把上次进程遗留的 running 改回 queued（保留 attempts），可重复调用。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"

JOB_STATUSES = frozenset({QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED})
TERMINAL_STATUSES = frozenset({SUCCEEDED, FAILED, CANCELLED})

INGEST_JOB_TYPE = "ingest"
BUSY_TIMEOUT_MS = 3000
MAX_ERROR_LEN = 2000
MAX_LIST_LIMIT = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    next_run_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, next_run_at, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""


def _utcnow(now: datetime | None = None) -> datetime:
    """当前 UTC 时间；测试可注入固定时间，naive 时间按 UTC 解释。"""
    if now is None:
        return datetime.now(UTC)
    return now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)


def _ts(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(frozen=True, slots=True)
class Job:
    """任务行（不可变快照；每次查询返回新对象）。"""

    id: str
    type: str
    payload: dict[str, Any]
    status: str
    attempts: int
    max_attempts: int
    error: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    next_run_at: datetime | None = None


def _row_to_job(row: sqlite3.Row) -> Job:
    created_at = _parse_ts(row["created_at"])
    updated_at = _parse_ts(row["updated_at"])
    if created_at is None or updated_at is None:
        raise ValueError(f"job {row['id']} has invalid timestamps")
    return Job(
        id=row["id"],
        type=row["type"],
        payload=json.loads(row["payload"]),
        status=row["status"],
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        error=row["error"],
        created_at=created_at,
        updated_at=updated_at,
        started_at=_parse_ts(row["started_at"]),
        finished_at=_parse_ts(row["finished_at"]),
        next_run_at=_parse_ts(row["next_run_at"]),
    )


class JobsStore:
    """SQLite jobs 表；所有方法线程安全（每次操作独立连接）。"""

    def __init__(
        self,
        db_path: str | Path,
        *,
        max_attempts: int = 3,
        busy_timeout_ms: int = BUSY_TIMEOUT_MS,
    ) -> None:
        self.db_path = Path(db_path)
        self.max_attempts = max(1, int(max_attempts))
        self.busy_timeout_ms = max(0, int(busy_timeout_ms))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            conn.commit()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=self.busy_timeout_ms / 1000)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        try:
            yield conn
        finally:
            conn.close()

    # ---------- 写入 ----------

    def enqueue(
        self,
        job_type: str,
        payload: Mapping[str, Any],
        *,
        max_attempts: int | None = None,
        now: datetime | None = None,
    ) -> Job:
        """插入一个 queued 任务；payload 以 JSON 持久化。"""
        if not job_type:
            raise ValueError("job_type 不能为空")
        if not isinstance(payload, Mapping):
            raise TypeError("payload 必须是 mapping")
        limit = self.max_attempts if max_attempts is None else int(max_attempts)
        if limit < 1:
            raise ValueError("max_attempts 必须 >= 1")
        timestamp = _utcnow(now)
        job_id = f"job_{uuid.uuid4().hex}"
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO jobs "
                "(id, type, payload, status, attempts, max_attempts, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 0, ?, NULL, ?, ?)",
                (
                    job_id,
                    job_type,
                    json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
                    QUEUED,
                    limit,
                    _ts(timestamp),
                    _ts(timestamp),
                ),
            )
            conn.commit()
        job = self.get(job_id)
        if job is None:  # pragma: no cover - 插入后立即查询必然存在
            raise RuntimeError(f"job {job_id} disappeared after insert")
        return job

    def claim_next(
        self,
        *,
        job_type: str | None = None,
        now: datetime | None = None,
    ) -> Job | None:
        """原子领取一个到期 queued 任务：置 running、attempts+1、写 started_at。"""
        timestamp = _utcnow(now)
        now_iso = _ts(timestamp)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                sql = "SELECT id FROM jobs WHERE status = ? AND (next_run_at IS NULL OR next_run_at <= ?)"
                params: list[Any] = [QUEUED, now_iso]
                if job_type is not None:
                    sql += " AND type = ?"
                    params.append(job_type)
                sql += " ORDER BY created_at ASC, rowid ASC LIMIT 1"
                row = conn.execute(sql, params).fetchone()
                if row is None or not self._mark_running(conn, row["id"], now_iso):
                    conn.rollback()
                    return None
                conn.commit()
                job_id = row["id"]
            except Exception:
                conn.rollback()
                raise
        return self.get(job_id)

    def mark_running(self, job_id: str, *, now: datetime | None = None) -> Job | None:
        """queued → running（attempts+1）；非 queued 不迁移，返回当前行。"""
        timestamp = _utcnow(now)
        now_iso = _ts(timestamp)
        with self._connection() as conn:
            self._mark_running(conn, job_id, now_iso)
            conn.commit()
        return self.get(job_id)

    @staticmethod
    def _mark_running(conn: sqlite3.Connection, job_id: str, now_iso: str | None) -> bool:
        cursor = conn.execute(
            "UPDATE jobs SET status = ?, attempts = attempts + 1, started_at = ?, updated_at = ?, "
            "finished_at = NULL, next_run_at = NULL WHERE id = ? AND status = ?",
            (RUNNING, now_iso, now_iso, job_id, QUEUED),
        )
        return cursor.rowcount == 1

    def mark_succeeded(self, job_id: str, *, now: datetime | None = None) -> Job | None:
        """running → succeeded（清 error / next_run_at）。"""
        timestamp = _utcnow(now)
        now_iso = _ts(timestamp)
        with self._connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = NULL, finished_at = ?, updated_at = ?, "
                "next_run_at = NULL WHERE id = ? AND status = ?",
                (SUCCEEDED, now_iso, now_iso, job_id, RUNNING),
            )
            conn.commit()
        return self.get(job_id)

    def mark_failed(
        self,
        job_id: str,
        error: str,
        *,
        now: datetime | None = None,
    ) -> Job | None:
        """running → failed（终止态，记录最后一次错误）。"""
        timestamp = _utcnow(now)
        now_iso = _ts(timestamp)
        message = str(error)[:MAX_ERROR_LEN]
        with self._connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ?, updated_at = ?, next_run_at = NULL "
                "WHERE id = ? AND status = ?",
                (FAILED, message, now_iso, now_iso, job_id, RUNNING),
            )
            conn.commit()
        return self.get(job_id)

    def mark_cancelled(self, job_id: str, *, now: datetime | None = None) -> Job | None:
        """queued/running → cancelled；已终止的任务保持不变。"""
        timestamp = _utcnow(now)
        now_iso = _ts(timestamp)
        with self._connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, updated_at = ?, next_run_at = NULL "
                "WHERE id = ? AND status IN (?, ?)",
                (CANCELLED, now_iso, now_iso, job_id, QUEUED, RUNNING),
            )
            conn.commit()
        return self.get(job_id)

    def schedule_retry(
        self,
        job_id: str,
        delay_s: float,
        error: str,
        *,
        now: datetime | None = None,
    ) -> Job | None:
        """running → queued，保留 attempts，next_run_at = now + delay_s（自动重试退避）。"""
        timestamp = _utcnow(now)
        next_run = timestamp + timedelta(seconds=max(0.0, float(delay_s)))
        message = str(error)[:MAX_ERROR_LEN]
        with self._connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, started_at = NULL, finished_at = NULL, "
                "next_run_at = ?, updated_at = ? WHERE id = ? AND status = ?",
                (QUEUED, message, _ts(next_run), _ts(timestamp), job_id, RUNNING),
            )
            conn.commit()
        return self.get(job_id)

    def retry(self, job_id: str, *, now: datetime | None = None) -> Job | None:
        """仅 failed → queued：重置 attempts/error/时间戳；其他状态返回 None。"""
        timestamp = _utcnow(now)
        now_iso = _ts(timestamp)
        with self._connection() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, attempts = 0, error = NULL, started_at = NULL, "
                "finished_at = NULL, next_run_at = NULL, updated_at = ? WHERE id = ? AND status = ?",
                (QUEUED, now_iso, job_id, FAILED),
            )
            conn.commit()
        return self.get(job_id) if cursor.rowcount == 1 else None

    def recover(self, *, now: datetime | None = None) -> int:
        """启动恢复：running → queued，保留 attempts；幂等，返回迁移行数。"""
        now_iso = _ts(_utcnow(now))
        with self._connection() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, started_at = NULL, finished_at = NULL, next_run_at = NULL, "
                "updated_at = ? WHERE status = ?",
                (QUEUED, now_iso, RUNNING),
            )
            conn.commit()
            return cursor.rowcount

    # ---------- 查询 ----------

    def get(self, job_id: str) -> Job | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row is not None else None

    def list(self, status: str | None = None, *, limit: int = 100) -> list[Job]:
        """按创建时间倒序列出任务；status 可选过滤。"""
        if status is not None and status not in JOB_STATUSES:
            raise ValueError(f"unknown job status: {status}")
        capped = max(1, min(int(limit), MAX_LIST_LIMIT))
        sql = "SELECT * FROM jobs"
        params: list[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(capped)
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_job(row) for row in rows]
