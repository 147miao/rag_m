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
from ..classifier.base import RouteDecision
from ..config import get_settings
from ..conversation.rewrite import Rewriter, SessionContext
from ..conversation.store import SessionStore
from ..ops.status import collect_status
from ..rag.llm import LLMClient
from ..rag.llm_router import build_llm
from .examples import examples_for
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
LOG_DIR = Path(__file__).resolve().parents[3] / "logs"


def _service(request: Request):
    return request.app.state.service


def _classifier(request: Request):
    return request.app.state.classifier


def _query_service(request: Request):
    """langgraph 后端依赖不可用时明确 503，除非显式开启 demo 回退。"""
    settings = get_settings()
    search = getattr(request.app.state, "search", None)
    if settings.rag_backend == "langgraph" and search is None and not settings.rag_allow_demo_fallback:
        raise HTTPException(
            status_code=503,
            detail=(
                "query unavailable: search service not ready (embedding model / milvus). "
                "Set RAG_ALLOW_DEMO_FALLBACK=true to allow demo answers."
            ),
        )
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="query service not ready")
    return service


def _classify_decision(request: Request, question: str) -> RouteDecision:
    return _classifier(request).classify(question)


def _classify(request: Request, question: str) -> ClassifierInfo:
    return ClassifierInfo(**_classify_decision(request, question).as_dict())


# ---------- multi-turn session helpers ----------
def _session_store(request: Request) -> SessionStore:
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        store = SessionStore(get_settings().session_store_path)
        request.app.state.session_store = store
    return store


def _rewriter(request: Request) -> Rewriter:
    """测试可注入 app.state.rewriter；默认用主 LLM 配置。"""
    rewriter = getattr(request.app.state, "rewriter", None)
    if rewriter is None:
        settings = get_settings()
        llm = LLMClient(settings.llm_base_url, settings.llm_model, settings.llm_api_key, settings.llm_temperature)
        rewriter = Rewriter(llm)
        request.app.state.rewriter = rewriter
    return rewriter


def _fill_session_fields(resp: QueryResponse, ctx: SessionContext | None) -> QueryResponse:
    if ctx is None:
        return resp
    resp.session_id = ctx.session_id
    resp.rewritten_question = ctx.rewritten_question
    resp.subquestions = list(ctx.subquestions)
    return resp


async def _session_context(req: QueryRequest, request: Request) -> SessionContext | None:
    """取历史 → 指代消解/子问题拆解；无 session_id 时返回 None（保持旧行为）。"""
    if not req.session_id:
        return None
    settings = get_settings()
    store = _session_store(request)
    record = await asyncio.to_thread(store.get_session, req.session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"session not found: {req.session_id}")
    history = await asyncio.to_thread(
        store.get_history, req.session_id, settings.session_max_turns, settings.session_max_chars
    )
    rewritten, subquestions = req.question, []
    if settings.session_rewrite_enabled and history:
        result = await asyncio.to_thread(_rewriter(request).rewrite, req.question, history)
        rewritten = str(result.get("rewritten") or req.question)
        subquestions = [str(item) for item in (result.get("subquestions") or []) if str(item).strip()]
    return SessionContext(
        session_id=req.session_id,
        question=req.question,
        rewritten_question=rewritten,
        subquestions=tuple(subquestions),
    )


async def _persist_turn(request: Request, ctx: SessionContext, resp: QueryResponse) -> None:
    """查询完成后写入用户原话与助手回答，供下一轮改写使用。"""
    store = _session_store(request)
    answer = resp.answer.lead
    if resp.answer.body:
        answer = f"{answer}\n\n{resp.answer.body}" if answer else resp.answer.body
    user_meta = {
        "rewritten_question": ctx.rewritten_question,
        "subquestions": list(ctx.subquestions),
    }
    assistant_meta = {
        "query_id": resp.id,
        "route": resp.route,
        "citations": list(resp.citations),
    }
    await asyncio.to_thread(store.append_message, ctx.session_id, "user", ctx.question, user_meta)
    await asyncio.to_thread(store.append_message, ctx.session_id, "assistant", answer, assistant_meta)


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


def _provider_health(llm) -> HealthCheck:
    """逐 provider 输出 ok/circuit_open；整体 ok 只看主 provider（状态里的第一个）。"""
    try:
        statuses = llm.provider_status() if llm is not None else {}
    except Exception as exc:  # noqa: BLE001 - health 不能因为 provider 状态异常而 500
        return HealthCheck(ok=False, detail=f"provider status error: {type(exc).__name__}: {str(exc)[:120]}")

    items = []
    for name, info in statuses.items():
        state = str((info or {}).get("state", "closed"))
        circuit_open = state == "open"
        items.append(
            {
                "name": name,
                "ok": not circuit_open,
                "circuit_open": circuit_open,
                "state": state,
                "failures": int((info or {}).get("failures", 0)),
            }
        )
    primary_ok = bool(items) and bool(items[0]["ok"])
    return HealthCheck(ok=primary_ok, detail=json.dumps(items, ensure_ascii=False, separators=(",", ":")))


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
    kg = getattr(request.app.state, "kg", None)
    checks["kg"] = HealthCheck(ok=kg is not None, detail=f"{len(kg.names)} entities" if kg is not None else "disabled")
    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        # 无 lifespan 的测试 / 早期请求：按当前 settings 懒构建一次。
        try:
            llm = build_llm(settings)
            request.app.state.llm = llm
        except Exception as exc:  # noqa: BLE001
            checks["providers"] = HealthCheck(
                ok=False, detail=f"provider init failed: {type(exc).__name__}: {str(exc)[:120]}"
            )
    if "providers" not in checks:
        checks["providers"] = _provider_health(llm)
    checks["langsmith"] = HealthCheck(
        ok=True,
        detail=f"enabled · {settings.langsmith_project}" if settings.langsmith_tracing and settings.langsmith_api_key else "disabled",
    )
    status = "ok" if all(c.ok for c in checks.values()) else "degraded"
    return HealthResponse(
        status=status,
        version=__version__,
        uptime_s=round(time.monotonic() - request.app.state.started_at, 1),
        checks=checks,
    )


# ---------- status ----------
@router.get("/readyz", summary="就绪检查（依赖可用才 200）")
async def readyz(request: Request) -> dict:
    svc = getattr(request.app.state, "search", None)
    if svc is None:
        raise HTTPException(status_code=503, detail={"ready": False, "reason": "search service unavailable"})
    try:
        rows = svc.store.count()
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"ready": False, "reason": f"milvus: {exc}"[:200]}) from exc
    return {"ready": True, "collection": svc.store.collection, "rows": rows}


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
@router.get("/examples", summary="示例问题（按当前语料库）")
async def examples() -> list[dict]:
    return examples_for(get_settings().milvus_collection)


@router.get("/classifier/metrics", summary="分类器评测指标")
async def classifier_metrics() -> dict:
    path = Path(__file__).resolve().parents[3] / "reports" / "classifier_metrics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"n": 0, "accuracy": None, "macro_f1": None, "note": "metrics unavailable"}


@router.post("/route", response_model=RouteResponse, summary="只做查询路由分类")
async def route(req: RouteRequest, request: Request) -> RouteResponse:
    return RouteResponse(**_classify(request, req.question).model_dump())


@router.post("/query", response_model=QueryResponse, summary="执行一次 RAG 查询")
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    service = _query_service(request)
    ctx = await _session_context(req, request)
    decision = None
    if req.route_override == "auto":
        routing_question = ctx.rewritten_question if ctx is not None else req.question
        decision = _classify_decision(request, routing_question)
        req = req.model_copy(update={"route_override": decision.route})
    if ctx is not None and getattr(service, "supports_session", False):
        resp = await service.query(req, decision=decision, session=ctx)
    else:
        resp = await service.query(req, decision=decision)
    if decision is not None:
        resp.classifier = ClassifierInfo(**decision.as_dict())
    _fill_session_fields(resp, ctx)
    if ctx is not None:
        await _persist_turn(request, ctx, resp)
    return resp


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@router.post("/query/stream", summary="SSE 流式查询（stage → token → done）")
async def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    service = _query_service(request)
    ctx = await _session_context(req, request)
    decision = None
    if req.route_override == "auto":
        routing_question = ctx.rewritten_question if ctx is not None else req.question
        decision = _classify_decision(request, routing_question)
        req = req.model_copy(update={"route_override": decision.route})

    async def gen():
        try:
            if ctx is not None and getattr(service, "supports_session", False):
                stream = service.stream(req, decision=decision, session=ctx)
            else:
                stream = service.stream(req, decision=decision)
            async for ev in stream:
                if isinstance(ev, StageEvent):
                    yield _sse("stage", ev.model_dump_json(by_alias=True))
                elif isinstance(ev, TokenEvent):
                    yield _sse("token", ev.model_dump_json(by_alias=True))
                elif isinstance(ev, DoneEvent):
                    _fill_session_fields(ev.response, ctx)
                    if ctx is not None:
                        await _persist_turn(request, ctx, ev.response)
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
    log_dir = LOG_DIR
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