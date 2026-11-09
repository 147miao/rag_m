"""v1.1 W4 异步任务队列测试：状态机、重试退避、恢复、retry 语义与 API。

全部使用临时 SQLite + fake ingest，不依赖 Milvus / GPU / 网络。
"""
from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ragm.api.main import create_app
from ragm.jobs.runner import DEFAULT_RETRY_DELAYS_S, JobRunner
from ragm.jobs.store import CANCELLED, FAILED, QUEUED, RUNNING, SUCCEEDED, JobsStore


@pytest.fixture
def store(tmp_path: Path) -> JobsStore:
    return JobsStore(tmp_path / "jobs.sqlite", max_attempts=3)


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_status_sequence_queued_running_succeeded(store: JobsStore) -> None:
    seen: dict[str, Any] = {}
    job = store.enqueue("ingest", {"src": "docs/a.md"})
    assert job.status == QUEUED
    assert job.attempts == 0

    def ingest(payload: dict[str, Any]) -> None:
        current = store.get(job.id)
        assert current is not None
        seen.update(status=current.status, attempts=current.attempts, payload=payload)

    runner = JobRunner(store, ingest_fn=ingest)
    done = runner.tick()
    assert done is not None
    assert done.status == SUCCEEDED
    assert done.attempts == 1

    # 执行期间任务已是 running（不是排队态）
    assert seen == {"status": RUNNING, "attempts": 1, "payload": {"src": "docs/a.md"}}
    saved = store.get(job.id)
    assert saved is not None
    assert saved.status == SUCCEEDED
    assert saved.attempts == 1
    assert saved.started_at is not None and saved.finished_at is not None
    assert saved.error is None


def test_tick_accepts_injected_executor(store: JobsStore) -> None:
    calls: list[dict[str, Any]] = []
    job = store.enqueue("ingest", {"src": "docs/b.md"})
    done = JobRunner(store).tick(ingest_fn=lambda payload: calls.append(payload))
    assert done is not None and job.id == done.id
    assert done.status == SUCCEEDED
    assert calls == [{"src": "docs/b.md"}]


def test_failure_retries_until_max_attempts_then_failed(store: JobsStore) -> None:
    calls: list[str] = []

    def boom(payload: dict[str, Any]) -> None:
        calls.append(str(payload["src"]))
        raise RuntimeError("boom")

    runner = JobRunner(store, ingest_fn=boom, retry_delays=(0.0, 0.0, 0.0))
    job = store.enqueue("ingest", {"src": "docs/fail.md"})

    first = runner.tick()
    second = runner.tick()
    for attempt, current in ((1, first), (2, second)):
        assert current is not None
        assert current.status == QUEUED
        assert current.attempts == attempt
        assert current.error is not None and "boom" in current.error
        assert current.next_run_at is not None

    final = runner.tick()
    assert final is not None
    assert final.status == FAILED
    assert final.attempts == 3
    assert final.error == "RuntimeError: boom"
    assert final.finished_at is not None
    assert len(calls) == 3
    assert runner.tick() is None
    saved = store.get(job.id)
    assert saved is not None and saved.status == FAILED


def test_retry_backoff_default_and_not_due_before_delay(store: JobsStore) -> None:
    def boom(payload: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    assert DEFAULT_RETRY_DELAYS_S == (5.0, 30.0, 120.0)
    runner = JobRunner(store, ingest_fn=boom, retry_delays=DEFAULT_RETRY_DELAYS_S)
    job = store.enqueue("ingest", {"src": "docs/backoff.md"})

    retried = runner.tick()
    assert retried is not None
    assert retried.status == QUEUED and retried.attempts == 1
    assert retried.next_run_at is not None
    remaining = (retried.next_run_at - datetime.now(UTC)).total_seconds()
    assert 3.0 < remaining <= 5.0
    # 5s 退避未到，下一次 tick 不应领取
    assert runner.tick() is None
    assert store.get(job.id).status == QUEUED  # type: ignore[union-attr]


def test_recover_requeues_running_and_keeps_attempts(store: JobsStore) -> None:
    job = store.enqueue("ingest", {"src": "docs/recover.md"})
    running = store.mark_running(job.id)
    assert running is not None
    assert running.status == RUNNING and running.attempts == 1

    assert store.recover() == 1
    recovered = store.get(job.id)
    assert recovered is not None
    assert recovered.status == QUEUED
    assert recovered.attempts == 1
    assert recovered.started_at is None
    assert recovered.next_run_at is None

    # 幂等：没有 running 任务时不再迁移
    assert store.recover() == 0


def test_recover_does_not_touch_terminal_jobs(store: JobsStore) -> None:
    done = store.enqueue("ingest", {"src": "docs/done.md"})
    store.mark_running(done.id)
    store.mark_succeeded(done.id)
    assert store.recover() == 0
    assert store.get(done.id).status == SUCCEEDED  # type: ignore[union-attr]


def test_retry_only_failed_and_resets_attempts(store: JobsStore) -> None:
    job = store.enqueue("ingest", {"src": "docs/retry.md"})
    assert store.retry(job.id) is None  # queued 不可手动重试

    store.mark_running(job.id)
    assert store.retry(job.id) is None  # running 不可手动重试

    store.mark_failed(job.id, "boom")
    retried = store.retry(job.id)
    assert retried is not None
    assert retried.status == QUEUED
    assert retried.attempts == 0
    assert retried.error is None
    assert retried.started_at is None and retried.finished_at is None

    assert store.retry(job.id) is None  # 已回到 queued


def test_cancelled_job_is_not_claimed(store: JobsStore) -> None:
    job = store.enqueue("ingest", {"src": "docs/cancel.md"})
    cancelled = store.mark_cancelled(job.id)
    assert cancelled is not None
    assert cancelled.status == CANCELLED
    assert store.claim_next() is None
    assert store.retry(job.id) is None
    assert store.get(job.id).status == CANCELLED  # type: ignore[union-attr]


def test_list_filters_status(store: JobsStore) -> None:
    failed = store.enqueue("ingest", {"src": "docs/f.md"})
    queued = store.enqueue("ingest", {"src": "docs/q.md"})
    store.mark_running(failed.id)
    store.mark_failed(failed.id, "boom")

    all_ids = {job.id for job in store.list()}
    assert all_ids == {failed.id, queued.id}
    failed_jobs = store.list(status=FAILED)
    assert [job.id for job in failed_jobs] == [failed.id]
    assert store.list(status=FAILED, limit=0)[0].id == failed.id
    with pytest.raises(ValueError):
        store.list(status="unknown")


def test_store_uses_wal_and_busy_timeout(store: JobsStore) -> None:
    assert store.busy_timeout_ms == 3000
    conn = sqlite3.connect(store.db_path)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


def test_default_ingest_script_is_loadable() -> None:
    from ragm.jobs.executor import _load_ingest_script

    script = _load_ingest_script()
    assert callable(script.build)
    assert callable(script.embed_chunks)


def test_lifespan_initializes_worker_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ragm.api import main
    from ragm.config import get_settings

    class FakeSearch:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def info(self) -> str:
            return "fake"

    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "lifespan-jobs.sqlite"))
    monkeypatch.setenv("JOBS_WORKER", "true")
    monkeypatch.setenv("RAG_BACKEND", "demo")
    monkeypatch.setenv("CLASSIFIER_BACKEND", "rules")
    monkeypatch.setattr(main, "SearchService", FakeSearch)
    get_settings.cache_clear()
    app = main.create_app()
    runner = None
    try:
        with TestClient(app) as client:
            assert client.get("/api/jobs").status_code == 200
            assert app.state.jobs_store is not None
            runner = app.state.jobs_runner
            assert runner is not None and runner.is_running
    finally:
        get_settings.cache_clear()
    assert runner is not None
    assert not runner.is_running
    assert app.state.jobs_store.db_path == tmp_path / "lifespan-jobs.sqlite"


def test_worker_thread_processes_job_and_shutdown_joins(store: JobsStore) -> None:
    processed = threading.Event()

    def ingest(payload: dict[str, Any]) -> None:
        processed.set()

    runner = JobRunner(store, ingest_fn=ingest, poll_interval=0.05)
    job = store.enqueue("ingest", {"src": "docs/worker.md"})
    runner.start()
    runner.wake()
    assert processed.wait(2.0)
    assert _wait_until(lambda: store.get(job.id) is not None and store.get(job.id).status == SUCCEEDED)
    runner.shutdown()
    assert not runner.is_running
    runner.shutdown()  # 幂等
    assert store.get(job.id).status == SUCCEEDED  # type: ignore[union-attr]


# ---------- API ----------


class FakeRunner:
    def __init__(self) -> None:
        self.wake_count = 0

    def wake(self) -> None:
        self.wake_count += 1


@pytest.fixture
def api_client(tmp_path: Path) -> Iterator[tuple[TestClient, JobsStore, FakeRunner]]:
    app = create_app()
    store = JobsStore(tmp_path / "api-jobs.sqlite", max_attempts=2)
    runner = FakeRunner()
    app.state.jobs_store = store
    app.state.jobs_runner = runner
    client = TestClient(app)
    try:
        yield client, store, runner
    finally:
        client.close()


def test_api_ingest_creates_queued_job_and_lists(
    api_client: tuple[TestClient, JobsStore, FakeRunner],
) -> None:
    client, store, runner = api_client
    response = client.post(
        "/api/ingest",
        json={"src": "docs/a.md", "collection": "rag_test", "recreate": True},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == QUEUED
    assert body["attempts"] == 0
    assert body["max_attempts"] == 2
    assert runner.wake_count == 1

    job_id = body["id"]
    detail = client.get(f"/api/jobs/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == job_id

    listing = client.get("/api/jobs")
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()["jobs"]] == [job_id]

    saved = store.get(job_id)
    assert saved is not None
    assert saved.payload == {"src": "docs/a.md", "collection": "rag_test", "recreate": True}


def test_api_job_not_found(api_client: tuple[TestClient, JobsStore, FakeRunner]) -> None:
    client, _store, _runner = api_client
    assert client.get("/api/jobs/job_missing").status_code == 404
    assert client.post("/api/jobs/job_missing/retry").status_code == 404


def test_api_retry_only_failed(api_client: tuple[TestClient, JobsStore, FakeRunner]) -> None:
    client, store, runner = api_client
    job = store.enqueue("ingest", {"src": "docs/retry-api.md"})

    queued = client.post(f"/api/jobs/{job.id}/retry")
    assert queued.status_code == 409

    store.mark_running(job.id)
    running = client.post(f"/api/jobs/{job.id}/retry")
    assert running.status_code == 409

    store.mark_failed(job.id, "boom")
    response = client.post(f"/api/jobs/{job.id}/retry")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == QUEUED
    assert body["attempts"] == 0
    assert body["error"] is None
    assert runner.wake_count == 1


def test_api_list_filters_failed(api_client: tuple[TestClient, JobsStore, FakeRunner]) -> None:
    client, store, _runner = api_client
    job = store.enqueue("ingest", {"src": "docs/failed-api.md"})
    store.mark_running(job.id)
    store.mark_failed(job.id, "boom")
    response = client.get("/api/jobs", params={"status": FAILED})
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["jobs"]] == [job.id]
    assert client.get("/api/jobs", params={"status": "bogus"}).status_code == 422


def test_api_end_to_end_worker_processes_ingest(tmp_path: Path) -> None:
    app = create_app()
    store = JobsStore(tmp_path / "e2e-jobs.sqlite", max_attempts=1)
    processed = threading.Event()

    def fake_ingest(payload: dict[str, Any]) -> None:
        processed.set()

    runner = JobRunner(store, ingest_fn=fake_ingest, poll_interval=0.05)
    app.state.jobs_store = store
    app.state.jobs_runner = runner
    client = TestClient(app)
    try:
        runner.start()
        response = client.post("/api/ingest", json={"src": "docs/e2e.md"})
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert processed.wait(2.0)
        assert _wait_until(
            lambda: client.get(f"/api/jobs/{job_id}").json()["status"] == SUCCEEDED
        )
        body = client.get(f"/api/jobs/{job_id}").json()
        assert body["attempts"] == 1
    finally:
        runner.shutdown()
        client.close()
