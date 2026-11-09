"""异步入库任务 API（v1.1 W4）。

POST /api/ingest 提交任务（202）；GET /api/jobs/{id} / GET /api/jobs 查询；
POST /api/jobs/{id}/retry 仅允许 failed 任务重试，其他状态 409。
SQLite 队列与 worker 由 ragm.jobs.store / ragm.jobs.runner 提供。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from ..jobs.store import FAILED, INGEST_JOB_TYPE, Job, JobsStore
from .schemas import IngestRequest, JobInfo, JobListResponse, JobStatus

router = APIRouter(prefix="/api", tags=["jobs"])


def _store(request: Request) -> JobsStore:
    store = getattr(request.app.state, "jobs_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="jobs store not initialized")
    return store


def _job_info(job: Job) -> JobInfo:
    return JobInfo(
        id=job.id,
        status=job.status,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _wake_runner(request: Request) -> None:
    runner = getattr(request.app.state, "jobs_runner", None)
    if runner is not None:
        runner.wake()


@router.post("/ingest", response_model=JobInfo, status_code=202, summary="提交异步入库任务")
def submit_ingest(req: IngestRequest, request: Request) -> JobInfo:
    job = _store(request).enqueue(
        INGEST_JOB_TYPE,
        {"src": req.src, "collection": req.collection, "recreate": req.recreate},
    )
    _wake_runner(request)
    return _job_info(job)


@router.get("/jobs/{job_id}", response_model=JobInfo, summary="任务详情")
def get_job(job_id: str, request: Request) -> JobInfo:
    job = _store(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return _job_info(job)


@router.get("/jobs", response_model=JobListResponse, summary="任务列表")
def list_jobs(
    request: Request,
    status: JobStatus | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> JobListResponse:
    jobs = _store(request).list(status=status, limit=limit)
    return JobListResponse(jobs=[_job_info(job) for job in jobs])


@router.post("/jobs/{job_id}/retry", response_model=JobInfo, summary="重试失败任务")
def retry_job(job_id: str, request: Request) -> JobInfo:
    store = _store(request)
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    if job.status != FAILED:
        raise HTTPException(
            status_code=409,
            detail=f"job {job_id} is {job.status}; only failed jobs can be retried",
        )
    retried = store.retry(job_id)
    if retried is None:
        raise HTTPException(status_code=409, detail=f"job {job_id} is no longer retryable")
    _wake_runner(request)
    return _job_info(retried)
