"""FastAPI 应用入口。

启动：
    uv run uvicorn ragm.api.main:app --host 127.0.0.1 --port 8000
或：
    make api
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..agent.agent_graph import AgentGraph
from ..agent.tools import ToolExecutor
from ..classifier.factory import build_classifier
from ..config import get_settings
from ..embeddings.reranker import build_reranker
from ..graph.kg import KGStore
from ..observability.tracing import configure_tracing
from ..rag.langgraph_service import LangGraphRagService
from ..rag.llm import LLMClient
from ..rag.service import DemoRagService
from ..store.search_service import SearchService
from .routes import router
from .schemas import ErrorBody, ErrorResponse

logger = logging.getLogger("ragm.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ROOT = Path(__file__).resolve().parents[3]
WEB_DIST = ROOT / "web" / "dist"
LOG_DIR = ROOT / "logs"


def _write_log(record: dict) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / "api.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("failed to write api log", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.started_at = time.monotonic()
    app.state.service = DemoRagService()
    app.state.classifier = build_classifier()
    app.state.search = None
    app.state.kg = None
    settings = get_settings()
    configure_tracing(settings)
    try:
        app.state.search = SearchService(settings)
        logger.info("search service ready: %s", app.state.search.info())
    except Exception:
        app.state.search = None
        logger.warning("search service unavailable (embedding model / milvus)", exc_info=True)
    app.state.kg = None
    if settings.rag_backend == "langgraph" and app.state.search is not None:
        try:
            kg = KGStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.kg_label, settings.kg_relation)
            n = kg.load()
            app.state.kg = kg
            logger.info("kg ready: %s entities", n)
        except Exception:
            app.state.kg = None
            logger.warning("neo4j unavailable, graph expansion disabled", exc_info=True)
        reranker = None
        if settings.rerank_enabled and settings.rerank_model_path:
            try:
                reranker = build_reranker(settings.rerank_model, str(settings.rerank_model_path))
                logger.info("reranker ready: %s", settings.rerank_model)
            except Exception:
                logger.warning("reranker unavailable", exc_info=True)
        llm = LLMClient(settings.llm_base_url, settings.llm_model, settings.llm_api_key, settings.llm_temperature)
        executor = ToolExecutor(app.state.search, app.state.kg, settings)
        agent = AgentGraph(
            llm,
            executor,
            max_iterations=settings.agent_max_iterations,
            max_tokens=settings.rag_max_tokens,
            max_calls=settings.agent_max_calls,
            total_tokens=settings.agent_max_tokens,
            retry=settings.agent_retry,
        )
        app.state.service = LangGraphRagService(
            app.state.classifier, app.state.search, llm, settings, kg=app.state.kg, agent=agent, executor=executor, reranker=reranker
        )
        logger.info("rag backend=langgraph | llm=%s | kg=%s | agent tools=%s", settings.llm_model, app.state.kg is not None, True)
    elif settings.rag_backend == "langgraph":
        if settings.rag_allow_demo_fallback:
            logger.warning(
                "rag backend=langgraph but search unavailable; serving demo because RAG_ALLOW_DEMO_FALLBACK=true"
            )
        else:
            logger.error("rag backend=langgraph but search unavailable; /api/query will return 503")
    else:
        logger.info("rag backend=demo | milvus=%s", settings.milvus_uri)
    yield
    logger.info("ragm api shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG-M API",
        version=__version__,
        description="路由式混合 RAG：查询分类 → 传统检索 / Graph RAG / Agentic RAG",
        lifespan=lifespan,
    )

    # 测试与无 lifespan 场景的默认状态
    app.state.started_at = time.monotonic()
    app.state.service = DemoRagService()
    app.state.classifier = build_classifier()
    app.state.search = None
    app.state.kg = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Elapsed-Ms"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Elapsed-Ms"] = str(elapsed_ms)
        if request.url.path.startswith("/api"):
            _write_log(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "elapsed_ms": elapsed_ms,
                }
            )
        return response

    app.include_router(router)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        body = ErrorResponse(
            error=ErrorBody(
                code=f"http_{exc.status_code}",
                message=str(exc.detail),
                request_id=getattr(request.state, "request_id", None),
            )
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        body = ErrorResponse(
            error=ErrorBody(
                code="validation_error",
                message=json.dumps(exc.errors(), ensure_ascii=False, default=str)[:800],
                request_id=getattr(request.state, "request_id", None),
            )
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    # 前端构建产物（存在则托管；API 路由优先注册，不会被静态文件吞掉）
    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
    else:

        @app.get("/", include_in_schema=False)
        async def root():
            return {
                "service": "RAG-M API",
                "version": __version__,
                "docs": "/docs",
                "web_dist": "not built (cd web && pnpm build)",
            }

    return app


app = create_app()