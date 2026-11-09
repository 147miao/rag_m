"""单 worker 线程任务执行器（v1.1 W4 异步入库）。

- tick() 领取并执行一个 queued 任务；执行函数可注入
  （测试用 fake ingest，不跑 GPU）；
- 失败按 5s / 30s / 120s 退避写 next_run_at，attempts 保留；达到 max_attempts 落 failed；
- start()/shutdown() 管理唯一的 worker 线程，start 幂等，shutdown join。
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any

from .executor import default_ingest
from .store import INGEST_JOB_TYPE, Job, JobsStore

logger = logging.getLogger(__name__)

IngestFn = Callable[[dict[str, Any]], Any]
DEFAULT_RETRY_DELAYS_S: tuple[float, ...] = (5.0, 30.0, 120.0)
DEFAULT_POLL_INTERVAL_S = 1.0


def _error_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text[:2000] if text else type(exc).__name__


class JobRunner:
    """单 worker 线程；也可在测试中直接调用 tick()。"""

    def __init__(
        self,
        store: JobsStore,
        ingest_fn: IngestFn | None = None,
        *,
        retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS_S,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        job_type: str | None = INGEST_JOB_TYPE,
    ) -> None:
        self.store = store
        self.retry_delays = tuple(max(0.0, float(delay)) for delay in retry_delays)
        self.poll_interval = max(0.05, float(poll_interval))
        self.job_type = job_type
        self._ingest_fn = ingest_fn
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """启动 worker（幂等）；lifespan 里根据 settings.jobs_worker 调用。"""
        if self.is_running:
            return
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(target=self._loop, name="ragm-jobs-worker", daemon=True)
        self._thread.start()
        logger.info("jobs worker started")

    def wake(self) -> None:
        """有新任务时唤醒 worker，避免等满 poll_interval。"""
        self._wake.set()

    def tick(self, ingest_fn: IngestFn | None = None) -> Job | None:
        """领取并执行一个任务；无可执行任务返回 None。"""
        job = self.store.claim_next(job_type=self.job_type)
        if job is None:
            return None
        executor = ingest_fn or self._ingest_fn or default_ingest
        try:
            executor(job.payload)
        except Exception as exc:  # noqa: BLE001 - 任务失败必须落库，不能杀死 worker
            error = _error_text(exc)
            delay = self._next_delay(job.attempts)
            if job.attempts < job.max_attempts and delay is not None:
                logger.warning(
                    "job %s failed (attempt %s/%s), retry in %.1fs: %s",
                    job.id,
                    job.attempts,
                    job.max_attempts,
                    delay,
                    error,
                )
                return self.store.schedule_retry(job.id, delay, error)
            logger.error("job %s failed permanently after %s attempt(s): %s", job.id, job.attempts, error)
            return self.store.mark_failed(job.id, error)
        logger.info("job %s succeeded", job.id)
        return self.store.mark_succeeded(job.id)

    def _next_delay(self, attempts: int) -> float | None:
        """第 attempts 次失败后的退避秒数；超出配置长度时复用最后一档。"""
        if not self.retry_delays:
            return None
        index = min(max(attempts - 1, 0), len(self.retry_delays) - 1)
        return self.retry_delays[index]

    def _loop(self) -> None:
        while not self._stop.is_set():
            job: Job | None = None
            try:
                job = self.tick()
            except Exception:  # 单次 tick 异常不能让 worker 线程退出
                logger.exception("jobs worker tick crashed")
            if self._stop.is_set():
                break
            # 有任务时立刻看下一个；空闲时等唤醒或轮询间隔。
            self._wake.wait(0.0 if job is not None else self.poll_interval)
            self._wake.clear()

    def shutdown(self, timeout: float = 5.0) -> None:
        """停止并 join worker 线程；可重复调用。"""
        thread = self._thread
        self._stop.set()
        self._wake.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))
            if thread.is_alive():
                logger.warning("jobs worker did not stop within %.1fs", timeout)
        self._thread = thread if thread is not None and thread.is_alive() else None
        logger.info("jobs worker stopped")
