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
from ..config import get_settings
from ..rag.service import DemoRagService
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
    settings = get_settings()
    logger.info(
        "ragm api ready | backend=demo | milvus=%s | llm=%s",
        settings.milvus_uri,
        settings.llm_model,
    )
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
