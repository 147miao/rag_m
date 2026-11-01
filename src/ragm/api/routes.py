"""HTTP 路由：/api/health · /api/scenarios · /api/query · /api/query/stream · /api/feedback"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .. import __version__
from ..config import get_settings
from ..ops.status import collect_status
from .schemas import (
    ClassifierInfo,
    DoneEvent,
    ErrorBody,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthCheck,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RouteRequest,
    RouteResponse,
    ScenarioSummary,
    SearchRequest,
    SearchResponse,
    StageEvent,
    TokenEvent,
)

router = APIRouter(prefix="/api", tags=["rag"])


def _service(request: Request):
    return request.app.state.service


def _classifier(request: Request):
    return request.app.state.classifier


def _classify(request: Request, question: str) -> ClassifierInfo:
    decision = _classifier(request).classify(question)
    return ClassifierInfo(**decision.as_dict())


# ---------- health ----------
async def _check_milvus(uri: str, timeout: float = 2.0) -> HealthCheck:
    def _ping() -> str:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=uri, timeout=timeout)
        try:
            collections = client.list_collections()
        finally:
            client.close()
        return f"{len(collections)} collections"

    try:
        detail = await asyncio.wait_for(asyncio.to_thread(_ping), timeout=timeout + 1)
        return HealthCheck(ok=True, detail=detail)
    except Exception as exc:  # noqa: BLE001 - 健康检查需要吞掉所有异常
        return HealthCheck(ok=False, detail=type(exc).__name__ + ": " + str(exc)[:120])


async def _check_ollama(base_url: str, model: str, timeout: float = 2.0) -> HealthCheck:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(base_url.rstrip("/") + "/api/tags")
            resp.raise_for_status()
            names = [m.get("name", "") for m in resp.json().get("models", [])]
        if model in names:
            return HealthCheck(ok=True, detail=f"{model} ready")
        return HealthCheck(ok=False, detail=f"model {model} not pulled (have {len(names)})")
    except Exception as exc:  # noqa: BLE001
        return HealthCheck(ok=False, detail=type(exc).__name__ + ": " + str(exc)[:120])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health(request: Request) -> HealthResponse:
    settings = get_settings()
    llm_root = settings.llm_base_url.removesuffix("/v1").rstrip("/")
    checks = dict(
        zip(
            ("milvus", "ollama"),
            await asyncio.gather(
                _check_milvus(settings.milvus_uri),
                _check_ollama(llm_root, settings.llm_model),
            ),
        )
    )
    try:
        info = _classifier(request).info()
        checks["classifier"] = HealthCheck(ok=True, detail=str(info.get("backend", "?")))
    except Exception as exc:  # noqa: BLE001
        checks["classifier"] = HealthCheck(ok=False, detail=type(exc).__name__)
    status = "ok" if all(c.ok for c in checks.values()) else "degraded"
    return HealthResponse(
        status=status,
        version=__version__,
        uptime_s=round(time.monotonic() - request.app.state.started_at, 1),
        checks=checks,
    )


# ---------- status ----------
@router.get("/status", summary="实验/任务进度")
async def status() -> dict:
    """读取训练/评测/数据进度（来自 scripts/status.py 同一采集器）。"""
    try:
        return collect_status()
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__, "detail": str(exc)[:300]}


# ---------- search ----------
@router.post("/search", response_model=SearchResponse, summary="混合检索（dense / BM25 / hybrid）")
async def search(req: SearchRequest, request: Request) -> SearchResponse:
    svc = getattr(request.app.state, "search", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="search service not ready (embedding model / milvus)")
    if req.mode == "small2big":
        res = await asyncio.to_thread(svc.search_small2big, req.query, req.k)
    else:
        res = await asyncio.to_thread(svc.search, req.query, req.k, req.mode)
    return SearchResponse(**res)


# ---------- scenarios ----------
@router.get("/scenarios", response_model=list[ScenarioSummary], summary="演示案例列表")
async def list_scenarios(request: Request) -> list[ScenarioSummary]:
    return _service(request).scenarios()


# ---------- query ----------
@router.post("/route", response_model=RouteResponse, summary="只做查询路由分类")
async def route(req: RouteRequest, request: Request) -> RouteResponse:
    return RouteResponse(**_classify(request, req.question).model_dump())


@router.post("/query", response_model=QueryResponse, summary="执行一次 RAG 查询")
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    classifier_info = None
    if req.route_override == "auto":
        classifier_info = _classify(request, req.question)
        req = req.model_copy(update={"route_override": classifier_info.route})
    resp = await _service(request).query(req)
    if classifier_info is not None:
        resp.classifier = classifier_info
    return resp


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@router.post("/query/stream", summary="SSE 流式查询（stage → token → done）")
async def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    async def gen():
        try:
            async for ev in _service(request).stream(req):
                if isinstance(ev, StageEvent):
                    yield _sse("stage", ev.model_dump_json(by_alias=True))
                elif isinstance(ev, TokenEvent):
                    yield _sse("token", ev.model_dump_json(by_alias=True))
                elif isinstance(ev, DoneEvent):
                    yield _sse("done", ev.model_dump_json(by_alias=True))
        except Exception as exc:  # noqa: BLE001 - SSE 内部错误以事件返回
            body = ErrorResponse(
                error=ErrorBody(code="stream_error", message=str(exc)[:300])
            )
            yield _sse("error", body.model_dump_json(by_alias=True))

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- feedback ----------
@router.post("/feedback", response_model=FeedbackResponse, summary="答案反馈")
async def feedback(req: FeedbackRequest, request: Request) -> FeedbackResponse:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fid = "fb_" + uuid.uuid4().hex[:10]
    record = {
        "id": fid,
        "query_id": req.query_id,
        "rating": req.rating,
        "comment": req.comment,
        "ts": datetime.now(UTC).isoformat(),
        "request_id": getattr(request.state, "request_id", None),
    }
    with (log_dir / "feedback.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return FeedbackResponse(ok=True, id=fid)