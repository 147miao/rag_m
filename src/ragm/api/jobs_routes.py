"""异步入库任务 API 契约骨架（v1.1 W4，owner: feat/jobs）。

prep 阶段只落路径/请求响应模型，全部返回 501；SQLite 队列与 worker 由 feature 分支实现。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .schemas import IngestRequest, JobInfo, JobListResponse

router = APIRouter(prefix="/api", tags=["jobs"])

_NOT_IMPLEMENTED = "jobs API not implemented yet (prep/v1.1 skeleton, owner: feat/jobs)"


def _todo() -> HTTPException:
    return HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.post("/ingest", response_model=JobInfo, status_code=202, summary="提交异步入库任务")
async def submit_ingest(req: IngestRequest) -> JobInfo:
    raise _todo()


@router.get("/jobs/{job_id}", response_model=JobInfo, summary="任务详情")
async def get_job(job_id: str) -> JobInfo:
    raise _todo()


@router.get("/jobs", response_model=JobListResponse, summary="任务列表")
async def list_jobs() -> JobListResponse:
    raise _todo()


@router.post("/jobs/{job_id}/retry", response_model=JobInfo, summary="重试失败任务")
async def retry_job(job_id: str) -> JobInfo:
    raise _todo()
